# CUDA 快速上手指南 —— 专为 Week 3 Tiling 学习准备

---

## 为什么要写这份指南

Week 3 的笔记里有很多 CUDA 代码。如果你从未写过 CUDA，这些代码看起来会很陌生：`<<<>>>` 是什么？`blockIdx` 和 `threadIdx` 有什么区别？`__global__` 和 `__shared__` 是干什么的？

这份指南的目标：**20 分钟内建立足够的 CUDA 知识，让你能逐行讲解 Week 3 笔记中的 naive GEMM 和 tiled GEMM 代码。**

不是全面的 CUDA 教程——只讲读懂 Week 3 代码需要的部分。

---

## 第一章：CUDA 编程模型 —— GPU 怎么组织线程

### 1.1 核心思想：成千上万个线程同时跑

CPU 编程：写一个循环，一次算一个数。

```c
// CPU 思维：一个线程，循环 N 次
for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}
```

CUDA 编程：启动 N 个线程，每个线程算一个数。

```cuda
// GPU 思维：N 个线程，每个线程算一个，没有循环
__global__ void add(int* A, int* B, int* C) {
    int i = threadIdx.x;  // 每个线程有自己的 ID
    C[i] = A[i] + B[i];
}
```

**GPU 的物理现实**：一个 RTX 4060 有 3072 个 CUDA Core，但你可以启动**百万**个线程。硬件以 32 个线程为一组（叫 **warp**）轮流调度到 CUDA Core 上执行。这意味着 GPU 设计上就假设你会启动海量线程——线程比核心多得多是正常的。

### 1.2 线程的三层组织：Grid → Block → Thread

这是理解 CUDA 代码最关键的概念。线程不是平铺的，而是分层的：

```
Grid (整个任务)
├── Block 0
│   ├── Thread (0,0)  Thread (0,1)  Thread (0,2)  ...
│   ├── Thread (1,0)  Thread (1,1)  Thread (1,2)  ...
│   └── Thread (2,0)  Thread (2,1)  Thread (2,2)  ...
├── Block 1
│   ├── Thread (0,0)  Thread (0,1)  Thread (0,2)  ...
│   ├── ...
└── Block 2, Block 3, ...
```

**为什么需要三层？** 因为 GPU 硬件就是这样设计的：

- 一个 Block 的所有线程在**同一个 SM（Streaming Multiprocessor）**上运行
- 同一个 Block 的线程可以共享 **Shared Memory**（你 Week 2 学过，~20 cycles 延迟）
- 不同 Block 的线程**不能**共享 Shared Memory，只能通过 Global Memory 通信（~400 cycles）
- 每个 Block 最多 1024 个线程（硬件限制）

**这直接决定了 tiling 的实现方式**：tiling 需要 Shared Memory 来缓存 tile 数据，所以一个 tile 必须由一个 Block 的线程来协作处理。

### 1.3 如何知道"我是谁"——四个内置变量

每个线程在运行时可以通过四个内置变量知道自己的位置：

```cuda
// 这些变量是 CUDA 自动注入的，不需要声明

blockIdx.x   // 我是哪个 Block？（Block 在 grid 中的列号）
blockIdx.y   // 我是哪个 Block？（Block 在 grid 中的行号）
blockIdx.z   // 我是哪个 Block？（Block 在 grid 中的深度号）

threadIdx.x  // 我是 Block 内的哪个线程？（列号）
threadIdx.y  // 我是 Block 内的哪个线程？（行号）
threadIdx.z  // 我是 Block 内的哪个线程？（深度号）

blockDim.x   // 一个 Block 有多少线程？（列数）
blockDim.y   // 一个 Block 有多少线程？（行数）
blockDim.z   // 一个 Block 有多少线程？（深度数）

gridDim.x    // 一个 Grid 有多少 Block？（列数）
gridDim.y    // 一个 Grid 有多少 Block？（行数）
```

**知道"我是谁"之后，就能算出"我负责哪个数据元素"**。这是每个 CUDA kernel 前几行代码在干的事。

### 1.4 全局索引计算公式（核心公式，必须记住）

```cuda
// 1D 场景：所有线程排成一条线
int global_id = blockIdx.x * blockDim.x + threadIdx.x;

// 2D 场景：线程和 Block 都是二维的（GEMM 里用的就是这个）
int row = blockIdx.y * blockDim.y + threadIdx.y;  // 全局行号
int col = blockIdx.x * blockDim.x + threadIdx.x;  // 全局列号
```
**`blockIdx.x * blockDim.x`（前面跳过了多少列）**
- `blockIdx.x`：代表当前线程所在的 Block，在整个 Grid 里的第几列（从 0 开始数）。
    
- `blockDim.x`：代表一个 Block 里面横向一共排了多少个线程。
    
- **含义**：在当前 Block 之前，已经有多少个线程列被前面的 Block 给占满了。这是当前 Block 的**起始列偏移量**。
**直观理解**：
```
假设 blockDim = (4, 4)，即每个 Block 有 4×4=16 个线程
blockIdx = (1, 0) 表示第二列第一个 Block

blockIdx.x * blockDim.x + threadIdx.x
= 1 * 4 + threadIdx.x
= 4 + threadIdx.x     ← 第二列 Block 的起始全局列号是 4
```

**在 Week 3 的 GEMM 代码中，这个公式出现两次**：

```cuda
// 在 naive GEMM 中（blockDim 是 (16, 16)）
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

// 在 tiled GEMM 中（blockDim 是 (TILE, TILE) = (32, 32)）
int row = blockIdx.y * TILE + threadIdx.y;
int col = blockIdx.x * TILE + threadIdx.x;
// 这两行完全等价，因为 blockDim.x = blockDim.y = TILE
```

### 1.5 Kernel 启动语法 `<<<>>>`

```cuda
// kernel_name<<<grid_size, block_size>>>(参数...);

// 具体例子：
dim3 threads(16, 16);  // 每个 Block 16×16=256 个线程
dim3 blocks(            // Grid 有 (N/16) × (M/16) 个 Block
    (N + 15) / 16,
    (M + 15) / 16
);
sgemm_naive<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
```

**`<<<blocks, threads>>>` 的含义**：启动 `blocks.x × blocks.y` 个 Block，每个 Block 有 `threads.x × threads.y` 个线程。总线程数 = 两个乘起来。

**为什么是 `(N + 15) / 16` 而不是 `N / 16`？** 这是向上取整的技巧。如果 N=1025，`1025/16 = 64.06`，用 `(N + 15) / 16 = 1040/16 = 65`。多出来的线程在 kernel 里通过 `if (row >= M || col >= N) return;` 来提前退出，不会访问越界。

---

## 第二章：CUDA 内存模型 —— 数据放在哪里

### 2.1 三种关键的内存类型

你的 Week 2 笔记已经详细讲解了物理存储层次。在代码层面，对应三种声明方式：

| 代码关键字 | 物理位置 | 延迟 | 作用范围 | 用途 |
|-----------|---------|------|---------|------|
| 无（默认） | Global Memory (VRAM/HBM) | ~400 cycles | 所有线程 | 输入输出的大矩阵 |
| `__shared__` | Shared Memory (SRAM) | ~20 cycles | 同一个 Block 内 | Tiling 的 tile 缓存 |
| 局部变量 | Register | ~1 cycle | 单个线程 | 累加器 `acc` |

### 2.2 代码示例：三种内存在 tiled GEMM 中的位置

```cuda
__global__ void sgemm_tiled(
    const float* __restrict__ A,  // ← Global Memory（VRAM 上）
    const float* __restrict__ B,  // ← Global Memory
    float* __restrict__ C         // ← Global Memory
) {
    __shared__ float As[TILE][TILE];  // ← Shared Memory（SM 的 SRAM 上）
    __shared__ float Bs[TILE][TILE];  // ← Shared Memory

    float acc = 0.0f;                 // ← Register（线程私有，最快）
    // ...
}
```

**数据移动路径**：

```
A (Global, ~400 cycles)
    → 协作加载 → As (Shared, ~20 cycles)
        → 累加 → acc (Register, ~1 cycle)
```

**这就是 tiling 的精髓在代码层面的体现**：数据只从慢的 Global Memory 搬一次到 Shared Memory，然后在 Shared Memory 上反复读取多次（每次 ~20 cycles 而不是 ~400 cycles）。

### 2.3 `__syncthreads()` —— 线程间的"栅栏"

这是 tiled kernel 中最关键的同步原语。

```cuda
__syncthreads();  // Block 内所有线程都执行到这里才能继续
```

**作用**：阻塞当前线程，直到 Block 内的**所有**线程都到达这个点。

**为什么需要两个 `__syncthreads()`？** 你的 Week 3 笔记第 4.4 节讲得非常清楚，这里从代码角度再补充一下：

```cuda
for (int t = 0; t < num_tiles; t++) {
    // 阶段 A: 从 Global Memory 加载数据到 Shared Memory
    As[threadIdx.y][threadIdx.x] = A[row * K + a_col];  // 线程各自加载
    Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];  // 线程各自加载

    __syncthreads();  // ← 屏障 1: "所有人都加载完了，Shared Memory 数据完整了"

    // 阶段 B: 从 Shared Memory 读取数据做计算
    for (int k = 0; k < TILE; k++) {
        acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();  // ← 屏障 2: "所有人都用完了当前数据，可以覆盖了"
    // 下一轮循环会把新的数据写进 As 和 Bs，覆盖旧数据
}
```

**去掉屏障 1 的后果**：线程 A 加载慢，线程 B 加载快。线程 B 已经进入阶段 B，读到的 `As[threadIdx.y][k]` 可能是线程 A 还没写完的旧值/半写值 → 结果错误。

**去掉屏障 2 的后果**：线程 A 计算快，已经进入下一轮循环的阶段 A，开始写新的 `As[0][0]`。线程 B 计算慢，还在阶段 B 读 `As[0][0]`。读到的是下一轮的新数据，混入了错误 tile 的值 → 结果错误。

---

## 第三章：CUDA 编程环境 —— 从零配置到跑通代码

### 3.1 检查你的环境

你的 Windows PowerShell 中运行以下命令，确认环境已就绪：

```powershell
# 1. 确认有 NVIDIA 显卡和驱动
nvidia-smi
# 应该看到: Driver Version: 592.00, CUDA Version: 13.1, GPU: RTX 4060

# 2. 确认 CUDA Toolkit 已安装
nvcc --version
# 应该看到: Cuda compilation tools, release 12.x

# 3. 确认 Nsight Compute 可用（性能分析工具）
ncu --version
# 如果没有，去 NVIDIA 官网下载 Nsight Compute
```

**如果你的环境中 `nvcc` 不可用**：
1. 去 [NVIDIA CUDA Toolkit 下载页](https://developer.nvidia.com/cuda-downloads) 下载 Windows 版
2. 安装时选 "自定义安装"，确保勾选 CUDA → CUDA Visual Studio Integration（如果你用 VS Code/VS）
3. 安装后重启 PowerShell，运行 `nvcc --version` 验证

### 3.2 创建一个最小的 CUDA 程序（Hello World for CUDA）

这是一个 30 行的完整程序，用来验证环境可用：

```cuda
// hello_cuda.cu
#include <stdio.h>

// 这是运行在 GPU 上的函数（kernel）
__global__ void hello_from_gpu() {
    // 只让线程 0 打印，避免 256 条重复消息
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Hello from GPU! Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
    }
}

int main() {
    // 启动 4 个 Block，每个 64 个线程 → 共 256 个线程
    hello_from_gpu<<<4, 64>>>();

    // 等待 GPU 完成
    cudaDeviceSynchronize();

    printf("Hello from CPU!\n");
    return 0;
}
```

编译运行：

```powershell
nvcc -o hello_cuda.exe hello_cuda.cu
.\hello_cuda.exe
```

如果看到 `Hello from GPU!`，环境就 OK 了。

### 3.3 编译 Week 3 的 GEMM 代码

Week 3 笔记第六章有完整的 `tiled_gemm.cu`。保存为文件后：

```powershell
# 编译（-O3 是最高优化级别）
nvcc -O3 -o tiled_gemm.exe tiled_gemm.cu

# 运行
.\tiled_gemm.exe
```

### 3.4 用 Nsight Compute 做 profiling

```powershell
# Profile naive kernel（通常是第一个 kernel launch）
ncu --set full --launch-skip 0 --launch-count 1 -o naive_report .\tiled_gemm.exe

# Profile tiled kernel（通常是第二个 kernel launch）
ncu --set full --launch-skip 1 --launch-count 1 -o tiled_report .\tiled_gemm.exe
```

然后用 Nsight Compute 的 GUI 打开 `.ncu-rep` 文件查看结果。重点看 `dram__bytes_read.sum`（从 HBM 读了多少字节）——这是 tiling 效果的最直接证据。

---

## 第四章：逐行讲解 Naive GEMM Kernel

现在我们来逐行讲解 Week 3 笔记第四章的 naive kernel。这是你需要在学习时能口头讲解的代码。

```cuda
__global__ void sgemm_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
```

| 语法 | 含义 |
|------|------|
| `__global__` | 这是 GPU kernel 函数，从 CPU 端调用（通过 `<<<>>>`），在 GPU 上执行 |
| `__restrict__` | 编译器优化关键字，告诉编译器这几个指针指向不重叠的内存区域。在 GEMM 中 A、B、C 是独立分配的，所以可以用。去掉也能跑，但可能影响编译器做向量化优化 |

```cuda
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
```

**这行在干什么？** 计算"我是谁"→"我负责 C 矩阵的哪个元素"。

举个例子：假设 M=N=64，用 16×16 的 Block：
- 第一个 Block（blockIdx.x=0, blockIdx.y=0）处理 C[0..15][0..15]
- 第二个 Block（blockIdx.x=1, blockIdx.y=0）处理 C[0..15][16..31]
- Block 内的 thread (0,0) 负责该 tile 的 (row=0, col=0)
- Block 内的 thread (3,5) 负责该 tile 的 (row=3, col=5)

```cuda
    if (row >= M || col >= N) return;
```

**这行在干什么？** 边界保护。当 M、N 不能被 blockDim 整除时，会启动多出来的线程。这些线程不干活直接返回。这就是第 1.5 节提到的"向上取整的多余线程"。

```cuda
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}
```

**这行在干什么？** 向量点积。计算 C[row][col] = A 的第 row 行 · B 的第 col 列。

重点看访存模式：
- `A[row * K + k]`：行优先存储，`row * K` 是第 row 行的起始偏移。随着 k 增加，访问的是连续地址（A 同一行的相邻列）→ **硬件能做 coalesced load（合并访存），这是好的**
- `B[k * N + col]`：随着 k 增加，访问的是 `B[0][col]`, `B[1][col]`, `B[2][col]`... 每个地址间隔 N×4 字节 → **stride 很大，无法合并访存，这是 naive kernel 性能差的一个关键原因**

---

## 第五章：逐行讲解 Tiled GEMM Kernel

### 5.1 Shared Memory 的声明

```cuda
#define TILE 32

__global__ void sgemm_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[TILE][TILE];  // ← 这一行
    __shared__ float Bs[TILE][TILE];  // ← 和这一行是核心
```

**`__shared__` 做了什么？**

1. 在 SM 的 SRAM 上分配数组（不是 VRAM）
2. 这个数组被**整个 Block** 的所有线程共享
3. 物理延迟 ~20 cycles，vs Global Memory 的 ~400 cycles
4. 大小：TILE×TILE = 32×32 = 1024 个 float = 4KB。两个 tile = 8KB，远在 48KB 限制内

**关键理解**：`As` 和 `Bs` 不是每个线程各有一份。整个 Block（1024 个线程）共享这一份。这就是"协作"的含义——1024 个线程一起加载它，一起用它计算。

### 5.2 全局位置计算

```cuda
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
```

**为什么是 `* TILE` 而不是 `* blockDim.y`？** 因为 Block 的尺寸恰好等于 TILE 大小（`dim3 threads(TILE, TILE)`），所以 `blockDim.y = TILE`。这行的含义和 naive kernel 完全一样，只是 tile 大小不同。

**一个 Block 负责一个 TILE×TILE = 32×32 的 C 子块**，Block 内的 1024 个线程各负责这个子块中的一个元素。

### 5.3 累加器与 Tile 循环

```cuda
    float acc = 0.0f;
    int num_tiles = (K + TILE - 1) / TILE;
```

`num_tiles` 是 K 维要被切成多少块。K=1024，TILE=32 → num_tiles=32。也就是说，沿着 K 维要循环 32 次，每次处理一个 tile pair。

**为什么需要这个循环？** 因为 Shared Memory 只有 8KB，装不下整个 A 和 B（1024×1024×4 = 4MB）。只能分 32 次，一次搬一小块（32×32），把这一小块能做的计算都做完，再换下一块。

```cuda
    for (int t = 0; t < num_tiles; t++) {
```

**每次循环做什么？** 加载第 t 个 A tile（列方向上的第 t 块）和第 t 个 B tile（行方向上的第 t 块），然后用它们更新 C 的累加和。

### 5.4 协作加载 A Tile

```cuda
        int a_col = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
```

**这是整个 kernel 中最巧妙的设计之一。**

**问题**：A tile 有 32×32=1024 个 float 需要从 Global Memory 加载。怎么加载？

**答案**：Block 里有 32×32=1024 个线程，每个线程加载**一个**元素（通过 `threadIdx.x` 和 `threadIdx.y` 索引，每个线程的 `(threadIdx.y, threadIdx.x)` 对是唯一的）。

- `As[threadIdx.y][threadIdx.x]`：当前线程把数据放进 Shared Memory 的对应位置
- `A[row * K + a_col]`：`row` 是全局行号（固定），`a_col = t * TILE + threadIdx.x` 是当前 tile 的第 threadIdx.x 列
- 线程 `(threadIdx.y=2, threadIdx.x=5)` 加载的是 A tile 中第 2 行、第 5 列的元素

**注意 A tile 和 B tile 是转置关系**：
- A tile：行方向对应 C 的 M 维（`row`），列方向对应 K 维（`a_col`）
- B tile：行方向对应 K 维（`b_row`），列方向对应 C 的 N 维（`col`）

这就符合 GEMM 的定义：`C[i][j] += A[i][k] * B[k][j]`

### 5.5 协作加载 B Tile

```cuda
        int b_row = t * TILE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
```

和 A tile 的加载逻辑对称。`b_row = t * TILE + threadIdx.y` 是当前 B tile 的行号（沿 K 维）。

注意这里 `Bs[threadIdx.y][threadIdx.x]` 存储的是 B 的原始布局（不转置），所以后续计算用 `Bs[k][threadIdx.x]`（沿 K 维遍历行，同一列不同行）。

### 5.6 第一个 `__syncthreads()` + 内层累加

```cuda
        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
```

**这个 for 循环在做什么？** 计算当前 tile pair 对 `C[row][col]` 的贡献。

- `As[threadIdx.y][k]`：A tile 的第 threadIdx.y 行（固定，因为 `row` 不变）、第 k 列（沿 K 维，由循环遍历）
- `Bs[k][threadIdx.x]`：B tile 的第 k 行（沿 K 维）、第 threadIdx.x 列（固定，因为 `col` 不变）
- 乘加后写入 `acc`：**`acc` 在寄存器中，延迟 ~1 cycle**

**为什么这里是 `As[threadIdx.y][k]` 和 `Bs[k][threadIdx.x]`？**

对应数学公式：C[row][col] += Σ_k A[row][k_local] × B[k_local][col]，其中 k_local 的范围是 `t*TILE` 到 `(t+1)*TILE-1`，映射到 Shared Memory 的索引 0..TILE-1。

注意：同一个线程的 `threadIdx.y` 和 `threadIdx.x` 在整个循环中不变，只有 `k` 在变。所以：
- `As[threadIdx.y][k]`：读的是 Shared Memory 中**同一行**的不同元素（连续地址，无 bank conflict）
- `Bs[k][threadIdx.x]`：读的是 Shared Memory 中**同一列**的不同元素

### 5.7 第二个 `__syncthreads()` + 结果写回

```cuda
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
```

32 次 tile 循环结束后，`acc` 中存的是完整的 `C[row][col]`。写回 Global Memory。

---

## 第六章：Naive vs Tiled —— 从代码角度看的本质区别

把两个 kernel 的核心循环并排比较：

```cuda
// ============= NAIVE =============         // ============= TILED =============
float acc = 0.0f;                             float acc = 0.0f;
for (int k = 0; k < K; k++) {                 for (int t = 0; t < num_tiles; t++) {
    // ↓ 每次迭代: 2 次 Global Memory 读        // 协作加载: 1024 线程各搬 1 个 → 并行!
    acc += A[row*K+k]   // ~400 cycles             As[ty][tx] = A[row*K + a_col];  // 各1次Global
         * B[k*N+col];  // ~400 cycles             Bs[ty][tx] = B[b_row*N + col]; //   (并行)
    // CPU时间占比: 计算 0.1%, 等数据 99.9%         __syncthreads();
}                                                     for (int k = 0; k < TILE; k++) {
                                                          // ↓ 每次迭代: 2 次 Shared Memory 读
                                                          acc += As[ty][k]   // ~20 cycles
                                                               * Bs[k][tx];  // ~20 cycles
                                                      }
                                                      __syncthreads();
                                                  }
```

**一句话总结**：Naive 每次读数据要等 ~400 cycles。Tiled 把数据搬到 Shared Memory（一次 ~400 cycles），然后在 Shared Memory 上读 32 次（每次只要 ~20 cycles）。**读 32 次的代价从 32×400=12800 降到 400+32×20=1040，快了约 12 倍。**

---

## 第七章：看懂 Week 3 代码需要记住的速查表

### 7.1 CUDA 关键字速查

| 关键字 | 简记 | 在哪出现 |
|--------|------|---------|
| `__global__` | 标记 GPU kernel 函数 | 每个 kernel 函数定义前 |
| `__shared__` | 在 SRAM 上分配变量 | `As[TILE][TILE]`, `Bs[TILE][TILE]` |
| `__restrict__` | 编译器优化：指针不重叠 | kernel 参数 |
| `__syncthreads()` | Block 内所有线程同步 | tile 加载后、计算后（2 次） |

### 7.2 线程变量速查

| 变量 | 含义 | 范围 |
|------|------|------|
| `threadIdx.x` | 线程在 Block 内的列号 | 0 ~ blockDim.x-1 |
| `threadIdx.y` | 线程在 Block 内的行号 | 0 ~ blockDim.y-1 |
| `blockIdx.x` | Block 在 Grid 中的列号 | 0 ~ gridDim.x-1 |
| `blockIdx.y` | Block 在 Grid 中的行号 | 0 ~ gridDim.y-1 |
| `blockDim.x` | 每个 Block 的列数 | = TILE = 32 |
| `blockDim.y` | 每个 Block 的行数 | = TILE = 32 |

### 7.3 全局索引公式

```cuda
// 行优先的 2D 矩阵（GEMM 中用这个）
int row = blockIdx.y * blockDim.y + threadIdx.y;  // 全局行
int col = blockIdx.x * blockDim.x + threadIdx.x;  // 全局列

// 线性索引（用于 1D 访问）
int idx = row * width + col;
```

### 7.4 常用 API 速查

| API | 作用 |
|-----|------|
| `cudaMalloc(&ptr, size)` | 在 GPU VRAM 上分配内存 |
| `cudaMemcpy(dst, src, size, dir)` | CPU ↔ GPU 数据搬运 |
| `cudaMemcpyHostToDevice` | CPU → GPU |
| `cudaMemcpyDeviceToHost` | GPU → CPU |
| `cudaDeviceSynchronize()` | 等待 GPU 完成所有任务 |
| `cudaFree(ptr)` | 释放 GPU 内存 |
| `cudaEventCreate/Record/ElapsedTime` | GPU 计时 |

---

## 第八章：你的 RTX 4060 的关键数字（写进脑子里）

每次分析代码性能时，对照这些数字：

| 参数 | 值 | 用于什么分析 |
|------|-----|------------|
| CUDA Cores | 3072 | 计算 FP32 的算力 |
| FP32 算力 | ~15.1 TFLOPS | Roofline 模型的计算上限 |
| FP16 Tensor Core 算力 | ~60 TFLOPS (dense) / ~120 TFLOPS (sparse) | 后续用 Tensor Core 时的上限 |
| 显存带宽 | 272 GB/s | Roofline 模型的带宽上限 |
| 算术强度阈值 (FP32) | 55.5 FLOP/Byte | **低于此 = memory-bound** |
| 算术强度阈值 (FP16 TC sparse) | 441 FLOP/Byte | 低于此 = memory-bound |
| Shared Memory / SM | 48 KB (可配置为 100 KB) | Tiling tile 大小上限 |
| Max Threads / Block | 1024 | TILE² 不能超过 1024 |
| Max Threads / SM | 1536 | occupancy 分析 |
| Register File / SM | 256 KB (65536 × 4B) | 寄存器压力分析 |

**快速判断 memory-bound 还是 compute-bound**：
- 算术强度 < 55.5 FLOP/Byte → memory-bound（算力空转，等数据）
- 算术强度 > 55.5 FLOP/Byte → compute-bound（带宽闲着，算力吃满）

Naive GEMM 的算术强度约 0.27 FLOP/Byte → **极度的 memory-bound**（差了 200 倍）。

---

## 第九章：学习路线建议

按这个顺序，每一步都能对照 Week 3 笔记的对应章节：

### Step 1：读懂线程模型（15 分钟）
- 读本指南第一章
- 在纸上画一个 Grid → Block → Thread 的三层结构图
- 口述：blockIdx、threadIdx、blockDim 三个变量怎么算出一个线程的全局 row/col

### Step 2：跑通 Hello CUDA（10 分钟）
- 复制第三章的 `hello_cuda.cu`，编译运行
- 证明你的 CUDA 环境是好的
- 改一下 `<<<4, 64>>>` 里的数字，看看会打印什么

### Step 3：逐行讲 Naive GEMM（20 分钟）
- 打开 Week 3 笔记的 `sgemm_naive` 代码
- 对照本指南第四章，逐行说出每行代码在干什么
- 特别要能解释：`int row = blockIdx.y * blockDim.y + threadIdx.y;`
- 特别要能解释：为什么这个 kernel 是 memory-bound

### Step 4：逐行讲 Tiled GEMM（30 分钟）
- 打开 Week 3 笔记的 `sgemm_tiled` 代码
- 对照本指南第五章，逐行说出每行代码在干什么
- 特别要能解释：
  - Shared Memory 的声明和用途
  - 协作加载的机制（为什么 1024 个线程各搬 1 个而不是 1 个线程搬 1024 个）
  - 两个 `__syncthreads()` 各自保护什么
  - 内层 `for (int k = 0; k < TILE; k++)` 循环的访存模式

### Step 5：跑代码 + 调参（30 分钟）
- 编译运行 Week 3 笔记第 6.2 节的完整代码
- 把 TILE 改成 8、16、32、64，记录每次的 GFLOPS
- 用 `ncu` 看 `dram__bytes_read.sum` 在 naive 和 tiled 之间的差异

### Step 6：能讲解（30 分钟）
- 找个人（或者录音），用白板讲一遍：
  - Naive GEMM 为什么是 memory-bound
  - Tiling 怎么减少 Global Memory 访问
  - 两个 `__syncthreads()` 的作用
  - 协作加载为什么快
- 能回答 Week 3 笔记末尾的 7 个自测问题

---

## 附录：常见编译错误排查

| 错误信息 | 原因 | 解决 |
|---------|------|------|
| `nvcc: command not found` | CUDA Toolkit 没装或没在 PATH | 安装 CUDA Toolkit，或找到 nvcc.exe 的路径加到 PATH |
| `no kernel image available for execution` | 编译的架构和 GPU 架构不匹配 | 加 `-arch=sm_89`（RTX 4060 是 Ada Lovelace, sm_89） |
| `too many resources requested for launch` | Shared Memory 或寄存器用太多 | 减小 TILE_SIZE 或 blockDim |
| `identifier "__shared__" is undefined` | 文件后缀不是 `.cu` | CUDA 代码必须保存为 `.cu` 文件，不能是 `.cpp` |
| `cudaErrorInvalidConfiguration` | Block 线程数超过 1024 | TILE² 不能 > 1024（TILE=32 刚好 1024，TILE=64 就不行了） |
