# Week 3 学习笔记：矩阵分块 Tiling 与片上数据复用

---

## 前两周回顾：为什么要学 Tiling？知识从哪来回哪去

### 一、Week 1 回顾：LLM 推理的两个阶段

#### 1.1 Prefill 阶段——计算密集型

用户输入一段 prompt，比如 "请帮我写一篇关于AI的文章"，假设 tokenize 后得到 2048 个 token。Prefill 阶段**一次性**把所有 token 喂给模型。

核心计算是 Self-Attention 中的矩阵乘法：

```
Q = X × W_Q    # (2048, d_model) × (d_model, d_model) → (2048, d_model)
K = X × W_K    # 同上
V = X × W_V    # 同上

S = Q × K^T    # (2048, d_model) × (d_model, 2048) → (2048, 2048)
               # 这是一个大矩阵乘法！计算量 = 2 × 2048² × d_model FLOP

O = S × V      # (2048, 2048) × (2048, d_model) → (2048, d_model)
               # 又是一个大矩阵乘法！
```

这几次矩阵乘法的维度都很大（batch × seq_len × d_model）。以 2048 tokens、d_model=4096 为例：
- Q×K^T 的计算量约 2 × 2048² × 4096 ≈ **34 GFLOP**（这只是单层单头的一部分）

**GPU 算力是瓶颈**。H100 有约 1000 TFLOPS 的 FP16 算力，这类运算能把它吃满。我们称 Prefill 为**计算密集型（Compute-bound）**。

#### 1.2 Decode 阶段——访存密集型

Prefill 完成后，模型开始逐个 token 输出。每次 decode 只处理**一个**新 token，但需要读入**之前所有 token** 的 K 和 V（KV Cache）。

```
当前只有 1 个新 token:
Q_new = (1, d_model)                    # 就一行

S = Q_new × K_cache^T                   # (1, d_model) × (d_model, 2048) → (1, 2048)
                                        # 计算量 = 2 × 1 × 2048 × d_model ≈ 16 MFLOP
                                        # 非常小！

但 K_cache 是 (2048, d_model)，假设 d_model=4096:
K_cache 大小 = 2048 × 4096 × 2 bytes (FP16) = 16 MB
每层都要读一次，每 decode 一步都要重新读
```

**问题来了**：16 MFLOP 的计算，GPU 瞬间就能算完。但为了这 16 MFLOP 的计算，你得先从 HBM 把 16 MB 的 KV Cache 读进来。HBM 带宽约 2 TB/s，读 16 MB 需要约 8 μs。而计算只需要约 16 ns——**99.8% 的时间在等数据**。

这就是 **访存密集型（Memory-bound）**：算力严重过剩，带宽是瓶颈。

#### 1.3 FlashAttention——Tiling 思想在 Attention 中的完美示范

标准的 Attention 计算有一个巨大的中间结果：

```python
# 标准实现（PyTorch 伪代码）
S = Q @ K.T          # S 的 shape: (N, N)，N=2048 时这是 4M 个元素
                      # S 被写回 HBM
P = softmax(S)        # P 也被写回 HBM
O = P @ V             # O 被写回 HBM
```

S 和 P 都是 **(N, N)** 的矩阵。N=2048 时是 16 MB（FP32），N=8192 时是 256 MB。每次 Attention 都要：
1. 把 S 写入 HBM
2. 下次计算 softmax 时再把 S 从 HBM 读回来
3. 把 P 写入 HBM
4. 下次计算 O 时再把 P 从 HBM 读回来

**FlashAttention 的解决办法**：不存 S 和 P。把 Q、K、V 都切成小块：

```
Q = [Q1, Q2, ..., Q_Tc]    切成 Tc 块，每块 (Br, d)
K = [K1, K2, ..., K_Tr]    切成 Tr 块，每块 (Bc, d)
V = [V1, V2, ..., V_Tr]    切成 Tr 块，每块 (Bc, d)

外层循环（遍历 K/V 的块）:
    把 Kj, Vj 从 HBM 加载到 SRAM

    内层循环（遍历 Q 的块）:
        把 Qi 从 HBM 加载到 SRAM
        在 SRAM 上计算 Sij = Qi × Kj^T         # 中间结果不写回 HBM！
        在 SRAM 上计算 Pij = softmax(Sij)       # 中间结果不写回 HBM！
        在 SRAM 上计算 Oi += Pij × Vj
        # 只把 Oi 写回 HBM

把 O 写回 HBM
```

**核心思想与本周 Tiling 完全一致**：
1. 数据切块，让每块能塞进 SRAM/Shared Memory
2. 在片上（on-chip）完成尽量多的计算
3. 中间结果不写回 HBM，只写最终结果

FlashAttention 省掉了 O(N²) 的 HBM 中间结果读写。当 N=8192 时，标准方法需要读写 ~256 MB 的 S/P 矩阵，FlashAttention 完全避免了这个开销。**这就是 tiling + 片上数据复用的力量**。

#### 1.4 知识链路

```
Week 1（FlashAttention）
    → 核心技巧：Tiling + Online Softmax + 片上计算
    → 本质：用 tiling 减少 HBM 访问

Week 2（GPU 存储层次）
    → 理解了 HBM、Shared Memory、Register 的速度差距
    → 建立了"把数据往快的存储搬"的直觉

Week 3（Tiling + 片上数据复用）
    → 把 FlashAttention 中用到的 tiling 思想剥离出来
    → 在更基础、更简单的场景（GEMM）中彻底搞懂它
    → 为后续更复杂的算子优化打底
```

---

### 二、Week 2 回顾：GPU 存储层次

#### 2.1 三层存储的物理位置与特性

```
                    ┌──────────────────────────────────────┐
                    │          GPU 芯片 (AD107)              │
                    │                                       │
   off-chip         │  ┌──────────────────────────────┐    │
   ──────────       │  │    Streaming Multiprocessor   │    │
   GDDR6 显存        │  │    (SM, 共 24 个)             │    │
   ┌────────┐       │  │                                │    │
   │ VRAM   │       │  │  ┌────────────────────────┐   │    │
   │ 8 GB   │       │  │  │  Register File         │   │    │
   │~400cyc │◄──────┼──┼──┤  每个 SM 65536×32-bit    │   │    │
   │272GB/s │       │  │  │  延迟: ~1 cycle          │   │    │
   └────────┘       │  │  │  线程私有                │   │    │
                    │  │  └────────────────────────┘   │    │
   ──────────       │  │                                │    │
                    │  │  ┌────────────────────────┐   │    │
                    │  │  │  Shared Memory / L1     │   │    │
                    │  │  │  每个 SM: 48KB (可配100KB)│   │    │
                    │  │  │  延迟: ~20-30 cycles     │   │    │
                    │  │  │  Block 内线程共享         │   │    │
                    │  │  └────────────────────────┘   │    │
                    │  └──────────────────────────────┘    │
                    │                                       │
                    │  ┌──────────────────────────────┐    │
                    │  │  L2 Cache (24 MB)            │    │
                    │  │  延迟: ~150-250 cycles        │    │
                    │  │  所有 24 个 SM 共享           │    │
                    │  └──────────────────────────────┘    │
                    └──────────────────────────────────────┘
```

关键数字（RTX 4060，Ada Lovelace 架构，AD107 核心）：

| 存储层                          | 大小                  | 带宽           | 延迟              | 作用范围          |
| ---------------------------- | ------------------- | ------------ | --------------- | ------------- |
| VRAM / Global Memory (GDDR6) | 8 GB                | **272 GB/s** | ~300-500 cycles | 所有 SM、所有线程    |
| L2 Cache                     | 24 MB               | ~1 TB/s      | ~150-250 cycles | 所有 24 个 SM 共享 |
| Shared Memory                | 48 KB/SM（可选 100 KB） | ~十几 TB/s     | ~20-30 cycles   | 一个 Block 内的线程 |
| Register File                | 256 KB/SM (64K×4B)  | 极高           | ~1 cycle        | 单个线程私有        |

**RTX 4060 算力补充**：

| 精度 | 峰值算力 | 来源 |
|------|---------|------|
| FP32 (CUDA Core) | **~15.1 TFLOPS** | 3072 CUDA Cores × 2 FMA × 2.46 GHz |
| FP16 Tensor Core (dense) | **~60 TFLOPS** | Ada 4th-gen Tensor Core, 不稀疏 |
| FP16 Tensor Core (sparse 2:1) | **~120 TFLOPS** | 结构化稀疏，2 倍 dense |

#### 2.2 这些数字意味着什么

一次 Global Memory 访问（~400 cycles）的时间里，GPU 可以执行约 400 次寄存器操作（~1 cycle 每次）。如果你的 kernel 频繁从 Global Memory 读取数据，那么大多数时间 SM 的算力单元（CUDA Core / Tensor Core）都在**空转等数据**。

Shared Memory 把延迟从 ~400 cycles 降到 ~20 cycles——快了约 **20 倍**。Register 更快，几乎不等待。

**Tiling 的目标**：把数据从 VRAM 搬到 Shared Memory/Register，通过片上数据复用提高算术强度，让 kernel 从 memory-bound 向 compute-bound 移动。

---

## Week 3 正篇：矩阵分块 Tiling 与片上数据复用

---

### 第三章：为什么需要 Tiling——Naive GEMM 的访存瓶颈

#### 3.1 Naive GEMM 的实现

目标：计算 C = A × B + C，其中 C(M×N), A(M×K), B(K×N)。

GPU 上最直观的实现：**一个线程负责 C 的一个元素**。

```cuda
// naive_sgemm.cu
__global__ void sgemm_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // 这个线程负责 C 的第 row 行、第 col 列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        // ← 每次循环体执行：
        //   从 Global Memory 读 A[row * K + k]    (~380 cycles)
        //   从 Global Memory 读 B[k * N + col]     (~380 cycles)
        //   做一次乘加                                   (~1 cycle)
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}
```

#### 3.2 逐行分析这个 Kernel 在 GPU 上到底发生了什么

**我们跟踪一个线程**，假设它负责计算 C[0][0]（row=0, col=0）。

这个线程需要 K 个 A 元素：A[0][0], A[0][1], ..., A[0][K-1] 和 K 个 B 元素：B[0][0], B[1][0], ..., B[K-1][0]。

```
迭代 k=0:  读 A[0][0]（Global Memory，~380 cycles）
           读 B[0][0]（Global Memory，~380 cycles）
           做 1 次 FMA (fused multiply-add)，约 1 cycle

迭代 k=1:  读 A[0][1]（Global Memory，~380 cycles）
           读 B[1][0]（Global Memory，~380 cycles）
           做 1 次 FMA，约 1 cycle

...

迭代 k=K-1: 读 A[0][K-1]（Global Memory，~380 cycles）
            读 B[K-1][0]（Global Memory，~380 cycles）
            做 1 次 FMA，约 1 cycle
```

**关键观察**：
- 每次循环 760 cycles 在等数据，1 cycle 在算
- 算力利用率 = 1 / 761 ≈ **0.13%**（实际不会这么差，因为有 L1/L2 cache 和 warp 调度帮忙隐藏延迟，但这说明了访存是瓶颈）

#### 3.3 从"全体数据"的角度看 Naive GEMM 的浪费

这不是一个线程的问题，放大到整个矩阵：

**A 的元素被重复读取的情况**：

A[0][0] 这个元素：
- 被线程(0,0) 读取，用来计算 C[0][0] = sum_k A[0][k] × B[k][0]
- 被线程(0,1) 读取，用来计算 C[0][1] = sum_k A[0][k] × B[k][1]
- 被线程(0,2) 读取，用来计算 C[0][2]
- ...
- 被线程(0, N-1) 读取，用来计算 C[0][N-1]

**A[0][0] 被 N 个线程各读了一次！** 如果每个线程都从 Global Memory 独立读取，就是 N 次 Global Memory 访问。

同理，**B[0][0] 被 M 个线程各读了一次**。

**总 Global Memory 读取量（最坏情况，无 cache）**：

```
A 的每个元素被读 N 次：M × K × N 次 float 读取
B 的每个元素被读 M 次：K × N × M 次 float 读取
C 写入一次：M × N 次 float 写入

总计 ≈ (2×M×N×K + M×N) × 4 bytes
```

当 M=N=K=4096 时：
- 总计算量 = 2 × 4096³ ≈ **137 GFLOP**
- 总访存量（最坏）≈ 2 × 4096³ × 4 bytes ≈ **512 GB**
- **算术强度** = 137 GFLOP / 512 GB ≈ **0.27 FLOP/Byte**

而你的 RTX 4060：
- 显存带宽 ≈ **272 GB/s**（128-bit GDDR6 @ 17 Gbps）
- FP32 算力 ≈ **15.1 TFLOPS**（3072 CUDA Cores × 2 FMA × 2.46 GHz）
- FP16 Tensor Core 算力（稀疏 2:1）≈ **120 TFLOPS**
- 当前代码用 FP32，要达到 compute-bound，需要算术强度 ≥ 15.1 TFLOPS / 272 GB/s ≈ **55.5 FLOP/Byte**
- 如果用 FP16 Tensor Core，需要算术强度 ≥ 120 TFLOPS / 272 GB/s ≈ **441 FLOP/Byte**

**0.27 FLOP/Byte vs 需要的 55.5 FLOP/Byte（FP32）或 441 FLOP/Byte（FP16 TC）**。这就是为什么 naive GEMM 是极度的 memory-bound：每次从显存读一个字节只能支撑 0.27 次浮点运算，而哪怕按 FP32 算，GPU 也需要每次读一个字节做 55.5 次运算才能吃满算力。**算力在疯狂等待数据**，差距约 200 倍。即使换成 FP16 Tensor Core，差距扩大到约 1600 倍——说明**不管什么精度，naive GEMM 都是毫无疑问的 memory-bound**。

#### 3.4 现实中没那么差——L1/L2 Cache 救了一部分

实际上 NVIDIA GPU 有 L2 cache（RTX 4060 上有 24 MB），同一 warp 的线程访问连续的 Global Memory 地址时会被合并（coalescing），这些硬件机制让 naive GEMM 的实际访存量远小于理论最坏值。但即便如此，naive GEMM 仍然是 memory-bound 的——cache 的容量有限，当矩阵大到一定程度时（比如 4096×4096 × 4 bytes = 64 MB 一个矩阵），单个矩阵就远超 cache 容量，cache 根本装不下。

**这正是 tiling 的出发点**：与其依赖硬件 cache 被动缓存，不如**显式地**把数据搬进 Shared Memory，由程序员控制复用策略。

---

### 第四章：Tiling 算法——怎么切、怎么算

#### 4.1 Tiling 的核心思路

矩阵太大 → 切成小块（Tiles）→ 每次只搬一块到 Shared Memory → 在 Shared Memory 上把这一块能做的计算都做完 → 再换下一块。

用图来表示：

```
矩阵 A (M×K)                  矩阵 B (K×N)                  矩阵 C (M×N)
┌─────────────┐               ┌─────────────┐               ┌─────────────┐
│ A00 A01 A02 │               │ B00 B01 B02 │               │ C00 C01 C02 │
│             │               │             │               │             │
│ A10 A11 A12 │               │ B10 B11 B12 │               │ C10 C11 C12 │
│             │               │             │               │             │
│ A20 A21 A22 │               │ B20 B21 B22 │               │ C20 C21 C22 │
└─────────────┘               └─────────────┘               └─────────────┘

每个子块大小：TILE × TILE

计算公式：C[i][j] = Σ_k A[i][k] × B[k][j]

其中 A[i][k] 是 A 的一个 tile，B[k][j] 是 B 的一个 tile。

例如：
C00 = A00·B00 + A01·B10 + A02·B20
```

**为什么这样就能减少 Global Memory 访问？**

考虑计算 C00：
1. 把 A00（TILE×TILE 个 float）从 Global Memory 搬进 Shared Memory。**只搬一次**。
2. 把 B00（TILE×TILE 个 float）从 Global Memory 搬进 Shared Memory。**只搬一次**。
3. 在 Shared Memory 上算 A00 × B00。这个 tile 内的计算涉及 TILE² 个线程，每个线程从 Shared Memory 读 TILE 次 A00 的元素和 TILE 次 B00 的元素。**Shared Memory 的读取不触发 Global Memory 访问**。
4. 再把 A01 和 B10 搬进 Shared Memory。同样各搬一次。
5. 以此类推。

**关键**：A00 里的 TILE² 个元素，从 Global Memory 读了 1 次，但在 Shared Memory 里被整个 block 的 TILE² 个线程反复使用了。

对比 naive：A00 里的每个元素从 Global Memory 被每个相关线程独立读了一遍（N 次）。

**从 N 次 → 1 次（或 K/TILE 次，因为一个 A tile 参与 K/TILE 个 tile 的计算），这就是 tiling 省的访存量。**

#### 4.2 一个具体的数字例子

设 M=N=K=4096，TILE=32：

**Naive**（无 cache）：
- 每个 A 元素被读 N=4096 次
- 每个 B 元素被读 M=4096 次

**Tiled**：
- A 有 (M/TILE) × (K/TILE) = (4096/32) × (4096/32) = 128 × 128 = 16384 个 tile
- 但实际上，每个 A tile 参与 N/TILE = 128 个 C tile 的计算（在列方向上）。
- 更准确地说：A 的一个 tile 在它所在的那 128 列 C tile 的计算中都会被用到。
- 每个 A 元素从 Global Memory 被读了 **N/TILE = 128 次**（每 tile 循环读一次）。

**减少倍数**：4096 / 128 = 32 = TILE_SIZE

**算术强度的变化**：

Tiled（TILE=32）的算术强度 ≈ Naive 的算术强度 × TILE_SIZE ≈ 0.27 × 32 ≈ **8.6 FLOP/Byte**。

**结论**：Block-level tiling（TILE=32）把算术强度从 0.27 提升到了 8.6 FLOP/Byte，改善了 32 倍，但离彻底 compute-bound 还远远不够。这就是为什么工业级实现（cuBLAS、CUTLASS）还要继续做 Warp-level tiling 和 Thread-level tiling——每一层 tiling 都是一次算术强度的倍增。只有三层 tiling 叠加，最终的算术强度才能跨过 Roofline 的拐点。

#### 4.3 Tiling Kernel 的完整结构

以下是 shared memory tiled GEMM 的完整 kernel，带逐段注释：

```cuda
#define TILE 32  // tile 大小：32×32

__global__ void sgemm_tiled(
    const float* __restrict__ A,   // M × K, row-major
    const float* __restrict__ B,   // K × N, row-major
    float* __restrict__ C,         // M × N, row-major
    int M, int N, int K
) {
    // ==========================================
    // 第一步：声明 Shared Memory
    // ==========================================
    // 这两个数组在 SM 的 SRAM 上分配，block 内所有线程共享
    // 物理上跟 Global Memory 是完全不同的硬件
    // TILE×TILE = 32×32 = 1024 个 float = 4KB per tile
    // 两个 tile = 8KB，在 48KB 的 shared memory 限制内绰绰有余
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // ==========================================
    // 第二步：计算每个线程负责的 C 元素位置
    // ==========================================
    // blockIdx 告诉我们处理哪个 C tile
    // threadIdx 告诉我们在 tile 内的位置
    int row = blockIdx.y * TILE + threadIdx.y;  // C 的全局行号
    int col = blockIdx.x * TILE + threadIdx.x;  // C 的全局列号

    // 这个线程的累加器（存在寄存器里，~1 cycle 延迟）
    float acc = 0.0f;

    // ==========================================
    // 第三步：遍历所有 A 和 B 的 tile
    // ==========================================
    // K 维被切成 (K + TILE - 1) / TILE 个 tile
    // 每轮循环处理一个 tile pair
    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++) {

        // ---- 3a. 协作加载 A tile ----
        // 每个线程从 Global Memory 加载 A 的一个元素
        // TILE×TILE = 1024 个线程，每个搬一个，一起完成整个 tile 的加载
        int a_col = t * TILE + threadIdx.x;  // 当前 A tile 内的列
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            // 越界（K 不能被 TILE 整除时）填 0
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // ---- 3b. 协作加载 B tile ----
        int b_row = t * TILE + threadIdx.y;  // 当前 B tile 内的行
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // ---- 3c. 第一个 __syncthreads() ----
        // 这行代码极其关键。作用是：
        // block 内的所有线程都执行到这里后，才能继续往下走。
        // 如果不加这个 barrier：
        //   有些线程还在从 Global Memory 加载数据（慢，几百 cycles），
        //   有些线程已经跑到第 3d 步开始读 Shared Memory 了，
        //   读到的可能是旧数据或半加载的数据 → 结果错误。
        __syncthreads();

        // ---- 3d. 在 Shared Memory 上累加 ----
        // 此时 As 和 Bs 的数据已经在 Shared Memory，
        // 读取延迟 ~20 cycles，远快于 Global Memory 的 ~380 cycles
        for (int k = 0; k < TILE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // ---- 3e. 第二个 __syncthreads() ----
        // 作用：确保 block 内所有线程都完成了第 3d 步的计算，
        // 没有人还需要当前 tile 的数据了，才能开始下一轮循环
        // 并用新数据覆盖 As 和 Bs。
        //
        // 如果不加这个 barrier：
        //   假设线程 A 算得快，已经进入下一轮循环（t+1），
        //   开始把新的 A tile 数据写入 As[0][0]。
        //   但线程 B 算得慢，还在第 3d 步读 As[0][0] 做累加。
        //   结果：线程 B 读到的是一半旧数据一半新数据的混合 → 结果错误。
        __syncthreads();
    }

    // ==========================================
    // 第四步：写回结果
    // ==========================================
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
```

> **[Bug Warning] 当前代码的硬件限制与工业库的解决办法**
>
> 上述 kernel 使用 **TILE² 个线程**（一个线程负责 C tile 的一个元素）。这里有一个重要的硬件约束：**CUDA 单个 Block 最多容纳 1024 个线程**。所以：
> - `TILE=32` → 线程数 = 32² = 1024 → 刚好擦线上限（安全）
> - `TILE=64` → 线程数 = 64² = 4096 → **超过 1024 线程上限，kernel 启动失败！**
>
> 这意味着**你不能简单地增大 TILE 来获得更多数据复用**——线程数限制把 Tile Size 卡在了 32 以下。但工业库（如 CUTLASS）却能使用 128×128 甚至 256×128 的 Block Tile。它们是怎么做到的？
>
> **答案：Thread-level Tiling（每个线程负责 C tile 中的多个元素，而非一个）**
>
> CUTLASS 的做法：
> ```
> Block Tile = 128×128（在 Shared Memory 中）
> ↓ 切成多个 Warp Tile
> Warp Tile = 64×64（一个 Warp 的 32 个线程协作处理）
> ↓ 每个线程用循环负责多个元素
> Thread Tile = 8×8（单线程负责 64 个 C 元素，用寄存器存 64 个累加器）
>
> Launch 参数: 每个 Block 只有 256 个线程（远低于 1024 上限）
> 协作加载: 256 个线程通过多次循环迭代把 128×128 的 Shared Memory tile 填满
>           每个线程在一次循环中搬运 float4（128-bit），循环 (128×128) / (256 × 4) = 16 次
> ```
>
> 具体来说：128×128 = 16384 个 float 需要从 Global Memory 搬到 Shared Memory。但只有 256 个线程。每个线程搬 16384 / 256 = 64 个 float。如果每次搬运 `float4`（4 个 float），每个线程循环 16 次就完成了。**协作加载的关键是并行度（256 个线程同时发 load 请求），而不是每个线程只 load 一次。**
>


---

> **[重要纠正] Block Size（TILE_SIZE）到底由什么决定？—— 不是一个条件，是三个**
>
> 上周汇报中被老师指正时，"block_size 能设多大，完全取决于 SRAM 有多大"。这个结论**不完整**。经过这一周的学习，这里给出完整的答案。
>
> **TILE_SIZE（即 block_size）由三个硬件资源共同约束，缺一不可：**
>
> | 约束条件 | 具体硬件限制（RTX 4060） | 如何影响 TILE_SIZE |
> |---------|----------------------|-------------------|
> | **① Shared Memory 容量** | 每个 Block 最多 48 KB（可选 100 KB）。两个 tile (As + Bs) 必须装得下：`2 × TILE² × 4 bytes ≤ shared_mem_per_block` | TILE ≤ √(48KB / 8B) ≈ 78。**这是最宽松的约束** |
> | **② 最大线程数** | CUDA 单个 Block 最多 **1024 个线程**。当前代码使用 `TILE²` 个线程 | TILE ≤ √1024 = 32。**这是当前代码写法下最紧的约束！** |
> | **③ Register File 容量** | 每个 SM 有 65536 个 32-bit 寄存器。每个线程用掉的寄存器 × block 内线程数 ≤ 65536。一个线程的累加器 `acc`（~几个寄存器）+ 地址计算等，假设每个线程用 ~32 个寄存器 | `32 regs × TILE² ≤ 65536` → TILE ≤ 45。比线程数约束宽松，但也很重要 |
>
> **三者取最小值**，所以：
>
> ```
> TILE_MAX = min(受限于 Shared Memory, 受限于线程数上限, 受限于 Register)
>          = min(78, 32, 45)
>          = 32     ← 这就是为什么我们的代码 TILE 只能到 32
> ```
>
> **为什么说"SRAM 决定 block_size"是不完整的？**
>
> 1. 在本周的代码写法中，**真正卡住 TILE 的是第②条（1024 线程上限）**，不是 Shared Memory。TILE=32 只用 8KB Shared Memory，远没碰到 48KB 的上限。
> 2. 即使 Shared Memory 很大（比如 A100 有 164KB/SM），如果你的代码每个线程只算一个 C 元素（TILE² 个线程），TILE 仍然被 1024 线程上限卡在 32。
> 3. 反过来说，**CUTLASS 能用到 TILE=128，不是因为 Shared Memory 大，而是因为它用了 Thread-level Tiling**——128×128 的 block tile 只用 256 个线程（打破第②条约束），每个线程负责多个 C 元素（用更多寄存器，受第③条约束）。
>
> **所以正确的理解是**：Block Size 是 Shared Memory、线程上限、Register 三个约束的**三元博弈**。当你改变代码结构（比如引入 Thread-level Tiling），三个约束的权重会重新分配——线程数约束放松了，但 Register 压力增大了。最终的最优值由 Occupancy（见 §6.4）决定：在满足所有约束的前提下，让一个 SM 能塞进尽量多的活跃 warp，从而最大化延迟隐藏能力。

#### 4.4 两个 `__syncthreads()` 为什么一个都不能少—— 用一个具体场景说明

假设一个 block 有 2 个线程（实际是 32×32=1024 个，为方便用 2 个举例），TILE=2：

**场景：没有第一个 `__syncthreads()`**

```
时间线：
t=0: 线程0 开始加载 As[0][0]（Global Memory，慢）
     线程1 已经加载完 As[1][1]，直接进入第 3d 步开始累加
     线程1 读 As[0][0] → 读到的是旧数据（线程0还没加载完，As[0][0]还是上一轮的值）
     累加结束 → C 的值错了
```

第一个 `__syncthreads()` = "所有人都把数据搬完了，才能开始算。"

**场景：没有第二个 `__syncthreads()`**

```
时间线：
t=0: 线程0 算完了第 3d 步的累加，很快进入下一轮循环（t=1），开始加载新的 As[0][0]
     线程1 算得慢，还在第 3d 步读 As[0][0] 做累加
     线程0 已经写入了新的 As[0][0]
     线程1 读到的 As[0][0] 是被线程0覆盖后的新值（属于下一个 tile）
     累加结束 → 混入了下一轮的数据 → C 的值错了
```

第二个 `__syncthreads()` = "所有人都用完了当前 tile 的数据，才能覆盖它。"

#### 4.5 协作加载（Cooperative Loading）详解

为什么每个线程加载一个元素，而不是用一个线程循环加载整个 tile？

**方案 A（协作加载）**：TILE×TILE 个线程，每个加载一个元素。
```
TILE = 32，一个 block 有 32 × 32 = 1024 个线程
每个线程执行 1 次 global memory load
→ 1024 次 load 同时发出（一个 warp 的 32 次 load 如果地址连续还会合并）
→ 总耗时 ≈ 1 次 global load 的延迟（并行）
```

**方案 B（单线程循环加载）**：一个线程循环加载 TILE² 个元素。
```
一个线程需要执行 1024 次 global memory load
→ 1024 次 load 串行发出
→ 总耗时 ≈ 1024 × ~380 = 389120 cycles
```

**为什么必须并行加载**：GPU 延迟隐藏的核心机制就是并行。一个 block 内的 1024 个线程同时发出 load 请求，虽然每个请求都要等几百 cycles，但 1024 个请求的时间窗口重叠，实际感知的延迟接近于一次 load 的时间。

#### 4.6 Shared Memory Bank Conflict

Shared Memory 被划分为 32 个 bank（对应一个 warp 的 32 个线程）。每个 bank 每个 cycle 可以服务一个线程。如果同一个 warp 的多个线程同时访问同一个 bank 的不同地址（不是同一个地址，同一个地址会被广播），就会发生 **bank conflict**——这些访问被串行化。

**在 Tiled GEMM 中**：

```cuda
// 当前 kernel 的 Shared Memory 访问模式（TILE=32，threads(32,32)）：
// for (int k = 0; k < TILE; k++) {
//     acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
// }
//
// 分析（假设 warp 内 32 个线程有相同的 threadIdx.y，threadIdx.x 从 0 到 31）：
//
// As[threadIdx.y][k]：
//   同一 warp 的所有线程有相同的 threadIdx.y，k 在内层循环是固定的
//   → 32 个线程访问的是 Shared Memory 的**同一个地址**
//   → 硬件广播机制：一个 bank 服务完毕后广播给所有 32 个线程
//   → **无 Bank Conflict** ✓
//
// Bs[k][threadIdx.x]：
//   同一 warp 的 32 个线程，threadIdx.x 分别为 0,1,...,31
//   Bs 是行优先存储，Bs[k][0] 到 Bs[k][31] 是连续的内存地址
//   Bank 编号 = ((k * TILE + threadIdx.x) * 4) / 4 % 32 = (k * 32 + threadIdx.x) % 32
//            = threadIdx.x % 32（因为 k*32 是 32 的倍数）
//   32 个线程的 threadIdx.x 各不相同 → 刚好 32 个 bank 各一个
//   → **无 Bank Conflict** ✓
//
// ★ 但这只是巧合！因为 TILE=32 恰好让 Bs 的每一行天然对齐到 bank 边界。
//
// ● 什么情况下会产生 Bank Conflict？
//
// 场景：当你需要按**列**方向访问 Shared Memory 时。考虑以下变体：
//
//   假设循环顺序不同，某个 kernel 这样访问：
//   for (int i = 0; i < TILE; i++) {
//       sum += As[i][threadIdx.x];  // 同一 warp 的不同线程，threadIdx.x 不同
//   }
//
//   这里 32 个线程都访问 As[i][threadIdx.x]，连续列在同一行，行优先布局下：
//   As[0][0], As[0][1], ..., As[0][31] → 连续地址 → 无冲突
//   但如果是列访问：
//   As[0][k], As[1][k], ..., As[31][k] → 每个元素隔了 TILE=32 个 float
//   Bank 编号 = (row * 32 + k) % 32 = (row * 32) % 32 = 0（全部在 Bank 0！）
//   → **32-way Bank Conflict！** 32 个请求被串行化成 32 次
//
// ● 为什么在 tiled GEMM 中这是一个潜在问题？
//
//   当前 kernel 中 k 变量在**最内层循环**，每次迭代 k 是固定的，
//   线程访问的是 Bs[k][threadIdx.x]（不同列）→ 无冲突。
//
//   但如果将来你使用 Warp-level Tiling（第五章），同一个 warp 内的不同线程
//   可能需要沿着 K 维的不同位置读取数据（如不同线程负责不同 k），
//   此时就可能触发列方向的 Bank Conflict。
```

**Bank Conflict 的检测与修复**：

在 `ncu` profiling 中关注 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 指标，它记录了 Shared Memory Load 的 bank conflict 次数。

**实验预测**：

> TILE=32 时当前 kernel 的 `shared__bank_conflicts` 应该接近 **0**（访问模式恰好在 bank 边界对齐）。但如果你将 Shared Memory 声明改为 `__shared__ float As[TILE][TILE + 1]`（padding 一列），然后故意按列方向访问（模拟 warp-level tiling 会面临的场景），在 `ncu` 中会观察到 **有 padding 的版本 `shared__bank_conflicts` 显著低于无 padding 版本**，因为 `TILE+1=33` 打破了 stride=32 的倍数关系，使得原来全落在 Bank 0 的请求被分散到了不同 bank。

**一般解决方法**：`__shared__ float As[TILE][TILE + 1];`（padding 一列），破坏 stride 恰好等于 bank 数量的对齐。实际 CUTLASS 等库会用更复杂的 swizzling 布局（XOR-based 地址映射）来彻底解决所有 bank conflict 模式。

这一块本周做两个实验感受一下：① 跑 TILE=32 的 kernel，在 ncu 中确认 bank conflict 为 0；② 写一个故意按列访问的小 test kernel，对比 padding 前后的 `shared__bank_conflicts` 差异。详细的优化放到下周。

---

### 第五章：多级 Tiling——Block → Warp → Thread

#### 5.1 为什么需要多级 Tiling

第四章的 kernel 只做了一级 tiling（Global Memory → Shared Memory）。但 Shared Memory 本身也有 ~20 cycles 的延迟。Shared Memory 到 Register 还能再加速 ~20 倍。

所以工业级实现（cuBLAS、CUTLASS）使用**三层 tiling**：

```
┌─────────────────────────────────────────────────────┐
│  第三层: Thread Tile                                  │
│  数据: Register ← 从 Shared Memory 加载               │
│  Tile 大小: 8×8 或 4×4（每个线程）                     │
│  延迟: ~1 cycle                                       │
│  复用: 一个 thread tile 内的元素被该线程内部循环复用    │
├─────────────────────────────────────────────────────┤
│  第二层: Warp Tile                                    │
│  数据: Shared Memory ← 从 Register 协作加载            │
│  Tile 大小: 64×64 或 32×64（一个 Warp）                │
│  延迟: ~20 cycles                                     │
│  复用: warp 内的 32 个线程共享 tile 内的数据            │
├─────────────────────────────────────────────────────┤
│  第一层: Block Tile（本章重点）                         │
│  数据: Shared Memory ← 从 Global Memory 协作加载        │
│  Tile 大小: 256×128 或 128×128（一个 Thread Block）     │
│  延迟: ~380 cycles → 降到 ~20 cycles                   │
│  复用: block 内的所有线程共享 tile 内的数据             │
└─────────────────────────────────────────────────────┘
```

#### 5.2 第一层：Block Tile（Global → Shared，本章已详细学习）

Block tile 的选择不是随便定的，它同时受到 §4.3 中详述的**三个硬件约束**（Shared Memory 容量、1024 线程上限、Register File 容量）的交叉限制。这里总结选择范围的直觉：

| 约束维度                                     | 具体公式（当前代码写法，TILE² 个线程）                                   | 结论          |
| ---------------------------------------- | -------------------------------------------------------- | ----------- |
| **太小**（TILE < 16）                        | 数据复用次数不够，算术强度接近 naive                                    | 性能差         |
| **恰好在 32**                               | 线程数 1024，刚好擦硬件上限，8KB Shared Memory，余量充足                  | 当前代码的最优值    |
| **太大**（TILE > 32）                        | 线程数超 1024 → kernel 直接启动失败                                | 不改代码结构则无法增大 |
| **如果改用 Thread-level Tiling**（CUTLASS 做法） | TILE 可到 128-256，因为只开 256 线程，Shared Memory/Register 成为新瓶颈 | 工业库的可行区间    |

核心教训见 §4.3 末尾的 **[重要纠正]** 框：**Block Size 是 Shared Memory、线程上限、Register 三者的最小交集，不是 SRAM 一个量说了算。**

#### 5.3 第二层：Warp Tile（Shared → Register
Block tile（比如 128×128）内部再切成更小的 warp tile（比如 32×32）。一个 warp 的 32 个线程协作，把 warp tile 的部分数据从 Shared Memory 搬到寄存器。这样内层循环访问的都是寄存器，延迟从 ~20 cycles 降到 ~1 cycle。

**为什么需要这一层**：Block tile 虽然存在 Shared Memory 里比 Global Memory 快，但 ~20 cycles 的延迟累积起来仍然可观。如果内层循环跑 32 次迭代，每次读 Shared Memory 要 ~20 cycles，一共 ~640 cycles。而如果把数据先搬进寄存器，每次迭代只要 ~1 cycle，32 次一共 ~32 cycles——**又快了 20 倍**。

#### 5.4 第三层：Thread Tile（寄存器编排）

每个线程不是只算 C 的一个元素，而是一次算一个 8×8 的小矩阵（64 个元素）。这样做的好处：
- 最大化寄存器利用率（每个线程的 64 个累加器存在不同的寄存器里）
- 配合 `float4` 向量化访存（一次 load 128-bit = 4 个 float）
- 减少指令发射开销（一次计算 64 个元素 vs 一次 1 个元素）
- 减少 Shared Memory 访问次数（一次 load 4 个 float，用 4 次，而不是 load 1 个用 1 次）

---

### 第六章：实践——从零跑通 Tiled GEMM 并验证

#### 6.1 环境确认

```powershell
# 以下命令在你的 Windows PowerShell 中运行

# 1. 确认有 NVIDIA 驱动
nvidia-smi
# 应看到: Driver Version: 592.00, CUDA Version: 13.1, GPU: RTX 4060

# 2. 确认 CUDA Toolkit 安装成功
nvcc --version
# 应看到: Cuda compilation tools, release 12.x

# 3. 确认 Nsight Compute 可用（profiling 工具）
ncu --version
# 如果没有，去 NVIDIA 官网下载 Nsight Compute
```

#### 6.2 完整可运行的代码


```cuda
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE 32

// ============================================================
// Kernel 1: Naive GEMM (baseline)
// ============================================================
__global__ void sgemm_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// ============================================================
// Kernel 2: Shared Memory Tiled GEMM
// ============================================================
__global__ void sgemm_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++) {
        // 协作加载 A tile
        int a_col = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // 协作加载 B tile
        int b_row = t * TILE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // 在 shared memory 上累加
        for (int k = 0; k < TILE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ============================================================
// 辅助函数：CPU 端参考实现（用于验证正确性）
// ============================================================
void sgemm_cpu(const float* A, const float* B, float* C,
               int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

// ============================================================
// 辅助函数：计时封装
// ============================================================
float time_kernel(void (*kernel)(), dim3 grid, dim3 block,
                  float* d_A, float* d_B, float* d_C,
                  int M, int N, int K, int warmup, int repeat) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup（让 GPU 进入稳定状态）
    for (int i = 0; i < warmup; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // 正式计时
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / repeat;  // 返回平均时间
}

// ============================================================
// 主函数
// ============================================================
int main() {
    // ---- 可调参数 ----
    int M = 1024, N = 1024, K = 1024;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // ---- 分配 GPU 内存 ----
    float *d_A, *d_B, *d_C_naive, *d_C_tiled;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C_naive, bytes_C);
    cudaMalloc(&d_C_tiled, bytes_C);

    // ---- 生成测试数据 ----
    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;  // 便于验证
    for (int i = 0; i < K * N; i++) h_B[i] = 2.0f;  // 便于验证

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    // ---- 设置 kernel launch 参数 ----
    dim3 threads_naive(16, 16);
    dim3 blocks_naive(
        (N + 15) / 16,
        (M + 15) / 16
    );

    dim3 threads_tiled(TILE, TILE);
    dim3 blocks_tiled(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    // ---- 运行并计时 ----
    float ms_naive = time_kernel(
        (void(*)())sgemm_naive, blocks_naive, threads_naive,
        d_A, d_B, d_C_naive, M, N, K, 5, 20
    );
    printf("Naive GEMM: %.3f ms\n", ms_naive);

    float ms_tiled = time_kernel(
        (void(*)())sgemm_tiled, blocks_tiled, threads_tiled,
        d_A, d_B, d_C_tiled, M, N, K, 5, 20
    );
    printf("Tiled GEMM: %.3f ms  (speedup: %.2fx)\n",
           ms_tiled, ms_naive / ms_tiled);

    // ---- 验证正确性 ----
    float *h_C_naive = (float*)malloc(bytes_C);
    float *h_C_tiled = (float*)malloc(bytes_C);
    float *h_C_ref   = (float*)malloc(bytes_C);

    cudaMemcpy(h_C_naive, d_C_naive, bytes_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_tiled, d_C_tiled, bytes_C, cudaMemcpyDeviceToHost);

    // 用 CPU 算一遍作为参考
    sgemm_cpu(h_A, h_B, h_C_ref, M, N, K);

    // 逐元素对比
    int errors_naive = 0, errors_tiled = 0;
    float expected = (float)K * 2.0f;  // 因为 A 全是 1.0，B 全是 2.0
    for (int i = 0; i < M * N; i++) {
        if (fabsf(h_C_naive[i] - expected) > 1e-2f) errors_naive++;
        if (fabsf(h_C_tiled[i] - expected) > 1e-2f) errors_tiled++;
    }
    printf("Naive errors: %d / %d\n", errors_naive, M * N);
    printf("Tiled errors: %d / %d\n", errors_tiled, M * N);

    // ---- 计算 GFLOPS ----
    float gflops = 2.0f * M * N * K / 1e9;
    printf("\n--- Performance ---\n");
    printf("GFLOPS (naive): %.1f\n", gflops / (ms_naive / 1000.0f));
    printf("GFLOPS (tiled): %.1f\n", gflops / (ms_tiled / 1000.0f));

    // ---- 清理 ----
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_naive); cudaFree(d_C_tiled);
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled); free(h_C_ref);

    return 0;
}
```


### 第七章：与 FlashAttention 的对比总结

| 维度                 | GEMM Tiling                                                     | FlashAttention Tiling                                      |
| ------------------ | --------------------------------------------------------------- | ---------------------------------------------------------- |
| **输入**             | A (M×K), B (K×N)                                                | Q (N×d), K (N×d), V (N×d)                                  |
| **中间结果**           | 部分累加和（简单加法）                                                     | softmax 分子分母（需要 online softmax 的 running max/sum，数值稳定性要求高） |
| **Tile 循环结构**      | 单层循环（沿 K 维切 tile）                                               | 双层循环（外层沿 K/V 切，内层沿 Q 切）                                    |
| **片上复用对象**         | A tile 和 B tile 的元素                                             | Q tile、K tile 的元素                                          |
| **省了什么 HBM 读写**    | A 和 B 的重复读取                                                     | S (QK^T) 和 P (softmax(QK^T)) 的整块 HBM 写入和回读（大小 O(N²)）       |
| **Tiling + 复用的本质** | 把 K 维切成 tile，每块在 shared memory 里算完，中间不访问 HBM                    | 把 N 维切成 tile，每块在 SRAM 里算完 QK^T → softmax → ×V，中间不访问 HBM    |
| **共享核心技术**         | Shared Memory / SRAM 上的协作加载 + `__syncthreads()` 护栏 + tile 内数据复用 | 完全相同的思路，只是多了 online softmax 的数值技巧                          |

**一句话归纳两者的关系**：GEMM Tiling 是基本招式，FlashAttention 是把这个招式用在了更复杂的场景（Attention）上，附加了 online softmax 的数值技巧。学懂了 GEMM Tiling，FlashAttention 的底层逻辑你就懂了 80%。

---

### 第八章：进阶——Warp-level Tiling 与 Register 数据复用


#### 8.1 回顾：当前 Kernel 的瓶颈在哪

在第四到六章中，我们实现了 Shared Memory tiled GEMM。它的数据流是：

```
外层循环 (t = 0..num_tiles):
    Global Memory ──load──→ Shared Memory (As, Bs)    ← ~400 cycles
    __syncthreads()
    内层循环 (k = 0..TILE):                             ← ~20 cycles/次
        从 Shared Memory 读 As[threadIdx.y][k]          ← 仍在等！
        从 Shared Memory 读 Bs[k][threadIdx.x]          ← 仍在等！
        Register 做乘加                                  ← ~1 cycle
    __syncthreads()
```

Block-level tiling 解决了 Global Memory 的延迟问题（从 ~400 cycles 降到 ~20 cycles）。但内层循环 **每次迭代仍然要从 Shared Memory 读数据**，`TILE` 次迭代累积的 Shared Memory 延迟是 `32 × 20 = 640 cycles`。

**如果能把数据从 Shared Memory 搬到 Register，内层循环就只访问 Register（~1 cycle）了。** 这就是 Warp-level Tiling 要做的事。

#### 8.2 核心思想：Block Tile 内再切一层 Warp Tile

```
Block Tile = 32×32（在 Shared Memory 中，当前代码）
     ↓ 切成 Warp Tile
每个 Warp Tile = 16×16（一个 Warp 的 32 个线程负责）
     ↓ 协作加载到 Register
每个线程从 Shared Memory 加载 Warp Tile 的一部分到自己的寄存器
     ↓ 在 Register 上做内层乘加
     ↓ ~1 cycle/次，总共 ~32 cycles（vs Shared Memory 的 ~640 cycles）
```

**具体的寄存器分配**（以 FP32 为例）：

每个线程负责 Warp Tile 内的一个 8×8 的 fragment（从 C 的角度看）。为了加速，线程先把 A 和 B 的 fragment 从 Shared Memory 搬到寄存器：

- 寄存器存 A_frag[8]：A 矩阵 Warp Tile 中该线程负责的 8 个元素
- 寄存器存 B_frag[8]：B 矩阵 Warp Tile 中该线程负责的 8 个元素
- 寄存器存 C_frag[8×8] = 64 个累加器：该线程负责的 C 的 8×8 子块

每个线程需要约 `8 + 8 + 64 = 80` 个寄存器。32 个线程 × 80 regs = 2560 regs / warp，远小于 65536 regs/SM。

#### 8.3 `float4` 向量化 Shared Memory Load

普通 Shared Memory 读：`float val = As[ty][k];` → 一次 load 一个 float（32-bit）

向量化读：`float4 val = ((float4*)&As[ty][k * 4])[0];` → 一次 load 四个 float（128-bit）

**为什么快 4 倍？** Shared Memory 的物理总线宽度是 128-bit。一次 `float` load 只用了 1/4 的带宽（带宽浪费 75%）。`float4` load 把一次 128-bit 事务用满。

在代码中：

```cuda
// 向量化从 Shared Memory 加载 A 的 fragment
// 一次 float4 load = 4 个 float，用满 128-bit 带宽
float4 a_frag = reinterpret_cast<float4*>(&As[threadIdx.y][k * 4])[0];
// a_frag.x, a_frag.y, a_frag.z, a_frag.w 就是 4 个连续元素

// 同样向量化加载 B 的 fragment
float4 b_frag = reinterpret_cast<float4*>(&Bs[k * 4][threadIdx.x])[0];
```

#### 8.4 完整代码结构（伪代码 + 关键片段）

```cuda
#define BLOCK_TILE 128   // Block tile: 保持在 Shared Memory
#define WARP_TILE  64    // Warp tile: Shared Memory → Register
#define THREAD_TILE 8    // Thread tile: 每个线程算 8×8 个 C 元素

__global__ void sgemm_warp_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // ---- Shared Memory（Block-level tile） ----
    __shared__ float As[BLOCK_TILE][BLOCK_TILE];
    __shared__ float Bs[BLOCK_TILE][BLOCK_TILE];

    // ---- 该线程在 C 中的全局位置 ----
    int c_row = blockIdx.y * BLOCK_TILE + /* warp 内偏移 + thread 内偏移 */;
    int c_col = blockIdx.x * BLOCK_TILE + /* warp 内偏移 + thread 内偏移 */;

    // ---- 寄存器（Thread-level fragment） ----
    // 64 个累加器：8×8 的 C fragment
    float c_frag[THREAD_TILE][THREAD_TILE] = {0.0f};

    // ---- 外层循环：遍历 K 维的 block tile ----
    for (int bk = 0; bk < K; bk += BLOCK_TILE) {
        // (1) 协作加载 Block Tile：Global Memory → Shared Memory
        //     与第四章的协作加载完全相同，但 tile 更大（128×128）
        //     256 个线程 × float4 × 多次循环 = 填满 128×128 shared memory
        load_block_tile_to_shared(A, B, As, Bs, bk);

        __syncthreads();

        // ---- 内层循环：遍历 Block Tile 内的 warp tile ----
        for (int wk = 0; wk < BLOCK_TILE; wk += WARP_TILE) {
            // (2) 协作加载 Warp Tile：Shared Memory → Register
            //     向量化 float4，每个线程搬一小段
            float a_frag[THREAD_TILE] = {0.0f};  // 存在寄存器里
            float b_frag[THREAD_TILE] = {0.0f};  // 存在寄存器里

            // float4 向量化：一次 load 128-bit
            for (int i = 0; i < THREAD_TILE / 4; i++) {
                float4 a_vec = reinterpret_cast<float4*>(
                    &As[warp_row + threadIdx.y][wk + i * 4]
                )[0];
                a_frag[i*4+0] = a_vec.x; a_frag[i*4+1] = a_vec.y;
                a_frag[i*4+2] = a_vec.z; a_frag[i*4+3] = a_vec.w;

                // B 同理...
            }

            // (3) 最内层：Register × Register → Register
            //     数据都在寄存器里了，延迟 ~1 cycle
            for (int ki = 0; ki < THREAD_TILE; ki++) {
                for (int row = 0; row < THREAD_TILE; row++) {
                    for (int col = 0; col < THREAD_TILE; col++) {
                        c_frag[row][col] += a_frag[ki] * b_frag[ki];
                    }
                }
            }
        }

        __syncthreads();
    }

    // ---- 写回 C（Register → Global Memory） ----
    for (int i = 0; i < THREAD_TILE; i++) {
        for (int j = 0; j < THREAD_TILE; j++) {
            if (c_row + i < M && c_col + j < N) {
                C[(c_row + i) * N + (c_col + j)] = c_frag[i][j];
            }
        }
    }
}
```

#### 8.5 与当前代码的关键差异对比

| 维度                | 当前代码（Block Tiling Only, §4） | 加上 Warp Tiling（本章）                       |
| ----------------- | --------------------------- | ---------------------------------------- |
| **Block Tile 大小** | TILE = 32 (受 1024 线程限制)     | BLOCK_TILE = 128（用 256 线程 + float4 循环加载） |
| **内层循环数据来源**      | Shared Memory（~20 cycles/次） | Register（~1 cycle/次）                     |
| **每个线程算 C 的元素数**  | 1 个                         | 64 个（THREAD_TILE = 8）                    |
| **寄存器用量**         | ~4 个（acc + 地址计算）            | ~80 个（8 A frag + 8 B frag + 64 C frag）   |
| **向量化**           | 无                           | float4（128-bit, 4× 带宽利用率）                |
| **三层 tiling**     | Block-level only            | Block → Warp → Thread，三层齐全               |

#### 8.6 寄存器压力与 Thread Tile 大小的权衡

每个线程负责的 THREAD_TILE 越大 → 寄存器中的 C fragment 越大 → 复用越多 → Shared Memory 访问越少。**但**：

- THREAD_TILE = 4（16 个 C 元素）：16 regs (C) + 4 regs (A) + 4 regs (B) ≈ 24 regs/thread → 很轻松
- THREAD_TILE = 8（64 个 C 元素）：64 + 8 + 8 ≈ 80 regs/thread → 适中
- THREAD_TILE = 16（256 个 C 元素）：256 + 16 + 16 ≈ 288 regs/thread → **寄存器不够，会 spilling！**

Register Spilling 发生时，编译器会把溢出到寄存器的数据存到 **local memory**（物理上在 HBM/VRAM）。这意味着你精心设计的数据搬到寄存器的优化，被编译器退化成 "从 HBM 读"——**性能反而暴跌**。

**用 `--ptxas-options=-v` 编译可以查看每个线程用了多少寄存器：**

```powershell
nvcc -O3 --ptxas-options=-v -o gemm.exe gemm.cu
# 输出: "Used 80 registers, 0 bytes spill stores"
# 出现 "xx bytes spill stores" → 寄存器溢出了，降低 THREAD_TILE
```

#### 8.7 性能预期

在你的 RTX 4060 上，对于 M=N=K=2048 的 FP32 SGEMM：

| Kernel 版本 | 预期 GFLOPS | 达到峰值 FP32 (15.1 TFLOPS) 的 |
|------------|------------|-------------------------------|
| Naive (§3) | ~50-100 | <1% |
| Block Tiling only (§4, TILE=32) | ~500-800 | ~3-5% |
| Block + Warp + Thread Tiling (§8) | ~2000-4000 | ~13-27% |
| cuBLAS SGEMM (参考上限) | ~8000-12000 | ~53-80% |

**从 5% → 27% 这一步，就是 Warp-level + Thread-level tiling 和 float4 向量化的贡献。**

cuBLAS 比你的手写 kernel 还快 2-3 倍，是因为它额外做了：Tensor Core（FP16/HGEMM）、双缓冲（计算与访存重叠）、prefetch、手写 SASS 指令微调等——这些就是路线 B、C、D 的内容。

#### 8.8 本周验证清单（Warp Tiling 部分）

- [ ] 画出 Block Tile → Warp Tile → Thread Tile 三层的数据流图，标出每一层数据在哪个物理存储上
- [ ] 用 `--ptxas-options=-v` 查看 register 用量，确认没有 spill stores
- [ ] 对比 float vs float4 的 shared memory load 事务数（ncu 中 `l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum`）
- [ ] 在同一张图上画出 naive → block tiled → warp tiled 三条 GFLOPS 曲线的对比

---

### 推荐阅读材料（按阅读顺序）

| 序号 | 材料 | 读什么 | 什么时候读 |
|------|------|--------|-----------|
| 1 | [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) | CUDA 线程模型、内存模型、`<<<>>>` 语法 | 动手写代码之前 |
| 2 | [How to Optimize a CUDA Matmul Kernel — Simon Boehm](https://siboehm.com/articles/22/CUDA-MMM) | 从 naive 到 tiled 的逐步优化过程，每步都有代码和性能数据 | 写好 naive kernel 后，跟着他的步骤走一遍 |
| 3 | PMPP (Programming Massively Parallel Processors) 第 5 章 | 5.4 节 "Tiled Matrix Multiplication"，详细推导了 tiling 的访存量和边界条件检查 | 理解 tiling 的理论基础时 |
| 4 | [CUTLASS Efficient GEMM 文档 Part 1](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md) | 三层 tiling (Threadblock → Warp → Thread) 的抽象层次和命名约定 | 学完 block-level tiling 后，为下周 warp-level 做准备 |
| 5 | [CUDA C++ Programming Guide — Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory) | Shared memory 的物理组织、bank 结构、synchronization 语义 | 遇到 bank conflict 或 `__syncthreads()` 的问题时查阅 |
| 6 | [Roofline Model (Williams et al., 2009)](https://crd.lbl.gov/assets/pubs_presos/parlab08-roofline-talk.pdf) | 算术强度的定义、Roofline 图的画法、用 Roofline 分析 kernel 处于哪个区间 | 前 15 页，理解 memory-bound vs compute-bound 的判断依据 |

---

### 本周检验清单

学习完成后逐条自测：

1. **口算访存量**：M=N=K=2048，TILE=32。Naive 下每个 A 元素被读了多少次？Tiled 下每个 A 元素从 Global Memory 被读了多少次？省了多少？
2. **解释 `__syncthreads()`**：为什么需要两个？分别保护的是什么？去掉其中一个分别会出什么错误？
3. **解释协作加载**：为什么 1024 个线程各搬一个元素比 1 个线程搬 1024 个元素快？
4. **解释 tile size 约束**：TILE=4 为什么慢？TILE=128 为什么也不行？受什么资源限制？
5. **解释 Occupancy**：Shared Memory 用量和 Occupancy 是什么关系？Occupancy 下降为什么会导致性能下降？
6. **解释 Profiling**：`ncu` 输出的 `dram__bytes_read.sum` 在 naive 和 tiled 两个 kernel 之间差了多少倍？为什么不是正好等于 TILE_SIZE？
7. **串联知识**：FlashAttention 的 tiling 和 GEMM tiling 的相同点和不同点各是什么？
