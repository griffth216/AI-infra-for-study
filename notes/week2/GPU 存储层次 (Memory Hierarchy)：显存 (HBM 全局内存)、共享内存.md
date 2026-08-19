# GPU 存储层次 (Memory Hierarchy)

## 零、引入：GPU 的核心矛盾 — 算得快 ≠ 搬得快

GPU 的设计目标是同时跑几千个线程，每个线程每个周期都在读写数据。然而有一个物理现实绕不过去：

> **内存越大就越慢，越快就越贵。一块芯片的面积和功耗是有上限的。**

以 RTX 4060 为例：

- **计算能力**：121 TFLOPS（FP16 Tensor Core），即每秒能执行 121 万亿次浮点运算
- **显存带宽**：272 GB/s（HBM/GDDR6 到计算核心的数据搬运速度）
- **矛盾量化**：如果所有数据都在全局显存上，每个 FP16 算 2 字节，272 GB/s ÷ 2 bytes = 每秒最多 1.36 × 10¹¹ 次操作。但 GPU 理论上每秒能算 1.21 × 10¹⁴ 次。

**算力利用率不到 0.2%。** 剩下的 99.8% 的时间，计算核心在等待数据送达。

> 这就是"内存墙"（Memory Wall）的核心：**算力过剩，带宽不足。**

用 **Arithmetic Intensity（算术强度）** 的概念来精确表达：

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Read from HBM}}$$

- RTX 4060：121 TFLOPS ÷ 272 GB/s ≈ **445 FLOPS/Byte**
- 这意味着：从 HBM 每读 1 个字节，需要做 445 次浮点运算，才能"喂饱"计算核心
- LLM 推理的 Decode 阶段，算术强度通常只有 **1-2 FLOPS/Byte** —— 远低于 445
- 结论：**Decode 阶段永远是 Memory-bound，算力大量浪费在等数据上**

H100 也有同样的矛盾：989 TFLOPS ÷ 3.35 TB/s ≈ **295 FLOPS/Byte**，算术强度不达标时一样 Memory-bound。这个比例从 Ampere 到 Blackwell 几乎没有本质变化——**硬件在涨，但算力和带宽的增速是同步的，矛盾不会自动消失。**

解决方案：多级存储层次。把常用的、马上要用的数据放在离计算核心更近、更快（但也更小）的存储器里。

---

## 一、全景：四层存储金字塔

一张图概括 GPU 的存储体系（以 A100 为例，覆盖 RTX 4060 到 H100 的完整跨度）：

### A100 全卡存储层次（最详细的参考基准）

```
  层级            每 SM 容量         总容量(全卡)        延迟             带宽(全卡)       相对 SRAM 延迟
  ───────────────────────────────────────────────────────────────────────────────────────────────
  寄存器           256 KB/SM          ≈ 27 MB(108SM)     ~0 cycle         最快(单cycle)      < 0.1×
    ↕
  SRAM/共享内存    192 KB/SM          ≈ 20 MB(108SM)     ~20 cycles        19 TB/s(SM内)      1×
    ↕
  L2 Cache           —               40 MB               ~200 cycles       4 TB/s             10×
    ↕
  HBM (HBM2e)        —               40 GB / 80 GB       ~600 cycles       2.0 TB/s           30×
```

**各层速度差距的本质原因**：

- **寄存器 → SRAM（1 cycle vs 20 cycles）**：寄存器在 ALU 旁边，连同一时钟周期内可完成读写。SRAM 虽然也在片上，但要走交叉开关（Crossbar）路由到 32 个 Bank，多了一跳。
- **SRAM → L2（20 vs 200 cycles）**：L2 在片上但离 SM 更远，且要服务 108 个 SM 的并发请求，需要仲裁和路由。
- **L2 → HBM（200 vs 600 cycles）**：HBM 在**芯片外面**（虽然很近，但不在同一块硅片上）。信号要经过硅中介层（Silicon Interposer）和微凸块，电信号传播延迟显著增大。

> 核心理念：**不是"喜欢"把数据往上层搬，是"不得不"搬——不搬计算核心就饿死了。**

---

## 二、第一层：寄存器 (Register File)

### 物理本质

寄存器是 GPU 芯片上**最接近计算单元（CUDA Core / Tensor Core）的存储单元**。每个 SM 内部包含一个巨大的寄存器文件（Register File），由 SRAM 单元（6T cell）组成，但访问速度比共享内存还快——因为每个时钟周期可以同时服务多个读写端口。

### A100 的寄存器文件架构

A100 的每个 SM 有 **4 个子分区（Sub-partitions）**，每个子分区有独立的：
- 16 个 FP32 CUDA Core
- 8 个 FP64 CUDA Core
- 1 个 Tensor Core
- 1 个 Warp Scheduler

**寄存器文件被均匀分配到 4 个子分区**，每个子分区有 **64 KB 寄存器文件**（16,384 个 32-bit 寄存器），合计每 SM 256 KB / 65,536 个 32-bit 寄存器。

### 关键数据

- **每 SM 寄存器容量**：256 KB（A100/H100/RTX 4060 类似规模）
- **每 SM 寄存器数量**：65,536 个 32-bit 寄存器
- **每线程最多寄存器数**：255 个（架构硬限制）
- **延迟**：≈ 0 cycle（计算和寄存器读写在同一个指令周期完成）
- **作用**：存**单线程的瞬时变量**

### 寄存器压力与 Occupancy 的权衡

这是 GPU 编程中最重要的资源约束之一。Occupancy（SM 占用率）的计算公式：

$$\text{Occupancy} = \frac{\text{Active Warps per SM}}{\text{Max Warps per SM}}$$

在 A100 上，每 SM 最多 **2048 个线程（64 个 Warp）**。但如果每个线程使用 255 个寄存器：

```
每线程 255 registers × 32 threads/Warp = 8,160 registers/Warp
每 SM 最多: 65,536 ÷ 8,160 ≈ 8 Warps
Occupancy = 8 / 64 = 12.5%
```

这意味着：**SM 上 87.5% 的时间没有足够的 Warp 可以切换，计算单元大量空闲。** 要达到 100% Occupancy（64 Warps），每线程最多只能用 32 个寄存器。

#### 为什么高 Occupancy 很重要？

GPU 通过**Warp 切换**来隐藏内存延迟。当一个 Warp 在等 HBM 数据（600 cycles），Scheduler 立即切换到另一个 Warp 执行。如果 SM 上活跃的 Warp 太少：

- HBM 访问延迟无法被隐藏 → 计算单元空闲
- 即使所有数据都在 SRAM 里，算力仍然吃不满

> 这就是 GPU 编程的核心矛盾之一：**用更多寄存器让单线程算得快 vs 用更少寄存器让更多 Warp 并存。** 编译器（nvcc）会自动做这个权衡，但你也可以通过 `__launch_bounds__` 提示编译器。

### 编程限制

- **线程私有**：线程 A 的寄存器线程 B 完全不可见
- **编译器管理**：不能显式声明"把这个变量放寄存器"。`nvcc` 自动分配，受 `--maxrregcount` 编译选项和 `__launch_bounds__` 提示的影响
- **寄存器溢出（Register Spilling）**：如果变量太多寄存器放不下，编译器会把部分变量"溢出"到 **L1 Cache（或 Local Memory，实际是 HBM 上的一块）**。这是性能杀手——本来 0 cycle 的存取变成了 ~600 cycles 的 HBM 访问

---

## 三、第二层：共享内存 (Shared Memory / SRAM)

> 如果说全局内存（HBM）是 GPU 园区外的"远端大仓库"，**共享内存就是流水线工人眼前的"高精尖工作台"**。

### 物理与架构属性

共享内存和 L1 Cache **共享同一块片上 SRAM 物理资源**。A100 上这个共享池的大小是 **192 KB/SM**，可由程序员配置：

- **L1 Cache 模式**：较多给 L1（硬件自动管理），较少给 Shared Memory
- **Shared Memory 模式（默认）**：最多 **164 KB** 可配置为 Shared Memory

配置方式（CUDA Runtime API）：
```c
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
```

或在 kernel 内直接声明：
```c
__global__ void kernel() {
    __shared__ float tile[4096];  // 4096 × 4 bytes = 16 KB
}
```

### 关键量化数据（各代 GPU）

| GPU | 每 SM Shared Memory 上限 | L1 + Shared Memory 总池 |
|-----|------------------------|------------------------|
| V100 (Volta) | 96 KB | 128 KB |
| A100 (Ampere) | 164 KB | 192 KB（1.7× V100） |
| H100 (Hopper) | 228 KB | 256 KB（1.33× A100） |
| RTX 4060 (Ada Lovelace) | 128 KB | 128 KB |
| B200 (Blackwell) | ~228 KB+ | ~256 KB+ |

### 三个核心用途

#### 用途一：分块计算 (Tiling)，减少 HBM 访问

以矩阵乘法 `C[M,N] = A[M,K] × B[K,N]` 为例。假设 `M=N=K=4096`：

- **不用 Shared Memory**：每个线程从 HBM 读取 A 的一行和 B 的一列。A 的每一行被 N 个线程各自读取一次 → 共 **N×K×N = 4096³ ≈ 680 亿次** FP16 元素读取（~136 GB）
- **使用 Shared Memory（Tile 32×32）**：每个 Block 先把 A 和 B 的一个 32×32 Tile 搬到 Shared Memory。同一 Block 内所有线程共享。A 的每个元素只从 HBM 读 **1 次**（而不是 N/32 次）→ 节省 **~30-100×** HBM 访问

> **Week1 关联**：`block_size=64` 就是 Tile 尺寸。外层循环锁定 Q 块在 SRAM 中，内层循环不断换入 K、V 块——本质就是把 HBM→SRAM 的搬运次数从 O(N²) 降到了 O(N)。

#### 用途二：线程间通信 (Inter-Thread Communication) + 同步

在做 Reduction（数组求和）时，同一个 Block 内的 256 个线程需要协作：

1. 每个线程从 HBM 读取一部分数据
2. 每个线程把局部结果写入 Shared Memory 的指定位置
3. `__syncthreads()` —— **阻塞点**，所有线程到达这一行才继续
4. 各线程从 Shared Memory 读取其他人的结果，做第二轮归约

`__syncthreads()` 是 GPU 编程的"集合哨"——确保**数据一致性**。没有它，某线程可能读到另一个线程还没写完的数据（数据竞争）。

#### 用途三：数据重组 / 挽救非合并访问

HBM 要求**连续的 32 个线程访问连续的 128 字节**（一个 cache line）才能达到峰值带宽。如果数据排布混乱（AoS → SoA 转换、图像转置、稀疏矩阵）：

1. 线程们先用合并访问的方式把乱序数据搬进 Shared Memory
2. 在 Shared Memory 里重新排列（Shared Memory 的随机访问很快，因为不经过 L2/HBM 的 cache line 机制）
3. 再以合并访问的方式写回 HBM

### 致命刺客：存储体冲突 (Bank Conflict)

这是共享内存优化的核心难点，也是 AI Infra 面试的高频考点。

#### 物理结构

Shared Memory 在硬件上被划分为 **32 个等大的存储体（Banks）**，以 4 字节（32-bit word）为单位交错编址：

```
Bank 编号 = (字节地址 / 4) % 32

例如:
  地址 0-3   (word 0)   → Bank 0
  地址 4-7   (word 1)   → Bank 1
  ...
  地址 124-127 (word 31) → Bank 31
  地址 128-131 (word 32) → Bank 0  ← 循环回来了
  地址 132-135 (word 33) → Bank 1
```

**每个 Bank 每个时钟周期只能处理一个请求。** 所有 Bank 可以同时工作，所以理论峰值是每周期 32 个请求。

#### Bank Conflict 的精确定义

- **无冲突（理想）**：一个 Warp 的 32 个线程访问 32 个**不同** Bank → 1 个事务完成
- **n 路冲突**：n 个线程访问**同一个 Bank 的不同地址** → 需要 n 个串行事务
- **广播（特殊豁免，CC 2.0+）**：多个线程访问**完全相同的地址** → 硬件检测到后**广播**给所有线程 → **不是冲突！**

#### 最常见的冲突模式：跨步访问（Strided Access）

```c
__shared__ float data[1024];
float val = data[threadIdx.x * stride];
```

- **`stride = 1`**（连续访问）：线程访问 word 0,1,2,...,31 → 32 个不同 Bank → **无冲突**
- **`stride = 2`**：线程访问 word 0,2,4,...,62。线程 0→word 0→Bank 0，线程 16→word 32→Bank 0 → **2 路冲突**
- **`stride = 32`**：全部线程访问 Bank 0（words 0,32,64,...,992）→ **32 路冲突**（最坏情况）

**通用规律**：如果 `gcd(32, stride) > 1`，就会有 Bank Conflict。只有 `stride` 为**奇数**时（`gcd(32, s) = 1`），一定无冲突。

#### 经典案例：2D 数组的列访问 — 以及 Padding 解法

```c
// 问题代码：列访问导致 32 路冲突
__shared__ float tile[32][32];  // 每行 32 个元素
float val = tile[threadIdx.x][col];
// 线程 0: tile[0][col] → word(0*32+col) → Bank (col % 32)
// 线程 1: tile[1][col] → word(1*32+col) = word(32+col) → Bank col → 和线程 0 同一个 Bank!
// 全部 32 个线程命中 Bank col → 32 路冲突!
```

**Padding 解法**：给每行加 1 个"假"元素：

```c
// 修复后：无冲突
__shared__ float tile[32][32 + 1];  // 每行 33 个元素，第 33 列是 padding
float val = tile[threadIdx.x][col];
// 线程 0: word(0*33+col), 线程 1: word(1*33+col)
// 33 % 32 = 1 → 每行的 col 元素 Bank 号错开 1 位 → 无冲突!
```

| 行号 | Word 地址（col=0） | Bank 编号 |
|------|-------------------|----------|
| 0 | 0 | 0 |
| 1 | 33 | 1 |
| 2 | 66 | 2 |
| ... | ... | ... |
| 31 | 1023 | 31 |

**这就是为什么 GEMM 的 CUDA 实现中经常看到 `__shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 1]` —— 多出来的 +1 不是写错了，是故意 Padding 来消除 Bank Conflict。** 性能改善通常在 5%-35%。

---

## 四、第三层：L2 Cache（二级缓存）

### 物理属性

L2 Cache 是 GPU 芯片上**所有 SM 共享**的最大一块片上 SRAM。它位于 SM 集群和 HBM 内存控制器之间，是数据进出 HBM 的"最后一关"。

| GPU | L2 Cache 大小 | 组织方式 |
|-----|-------------|---------|
| V100 (Volta, 2017) | 6 MB | 32 个切片 (slices) |
| A100 (Ampere, 2020) | **40 MB**（6.7× V100） | 80 个切片，每个 512 KB |
| H100 (Hopper, 2022) | **50 MB**（1.25× A100） | 更多切片 + 更高时钟 |
| RTX 4060 (Ada Lovelace) | 24 MB | 缩小的消费级规格 |
| B200 (Blackwell, 2025) | **更大**（双 Die） | 双 Die 各有独立 L2，合计 ~2× H100 |

### A100 L2 Cache 的架构细节

A100 的 40 MB L2 被划分为 **80 个切片（Slices）**，通过**分层交叉开关（Hierarchical Crossbar）**连接：

- **L2 带宽**：约 **4 TB/s**（是 HBM 带宽的 2 倍）
- **延迟**：约 200 cycles（~100ns @ 1.4GHz）
- **Cache Line 大小**：128 字节（一段连续地址，一个 L2 miss 会从 HBM 搬回一个完整的 cache line）
- **驻留控制 (L2 Residency Control)**：A100 支持**手动锁定部分 L2 空间**（最多 ~30 MB）。可以将反复访问的数据（如 KV Cache 的头几层）永久钉在 L2 中，不被自动驱逐。

### L2 的命中判定流程

```
线程请求数据 → 先查 L1（SM 内, ~20 cycles）
  ↓ miss
→ 查 L2（片上, ~200 cycles）
  ↓ miss
→ 查 HBM（片外, ~600 cycles）—— 这次访问的代价是 L1 的 30 倍
```

#### 为什么 L2 命中率对推理很重要？

LLM 推理的 Decode 阶段，**每生成一个新 token**：

1. 从 HBM 加载**整个模型权重**（几十 GB）→ 每个 token 都要！
2. 从 HBM 加载**全部 KV Cache**（几 GB）→ 逐 token 增长

如果 L2 能缓存住权重矩阵的一部分（比如第一层 Transformer 的 QKV 投影矩阵），第二个 token 开始就能命中 L2。PagedAttention 的分块设计也提升了 L2 效率——按固定 Block 存储 KV，每个 Block 的大小刚好匹配 L2 cache line 的整数倍。

### L2 vs Shared Memory 的关键区别

| | Shared Memory (SRAM) | L2 Cache |
|---|---|---|
| 谁管理 | **程序员手动控制**（`__shared__`） | **硬件自动管理**（LRU 或类似替换策略） |
| 可见范围 | 单个 Thread Block | 全 GPU 所有 SM |
| 编程代价 | 高（算容量、避 Bank Conflict、手写同步） | 零（自动透明） |
| 优化上限 | 极高（确定性，可精确到 cycle） | 受限于硬件替换策略 |
| 典型容量 | 128-256 KB/SM | 24-50 MB/全卡 |

> **AI Infra 工程师的核心技能：把 L2 自动做的事，用手动控制的 Shared Memory 做得更极致。FlashAttention 就是将 Attention 计算从"依赖 L2/HBM"变成"手动在 Shared Memory 上完成"。**

---

## 五、第四层：HBM (High Bandwidth Memory，高带宽内存)

> HBM 部分保留你已有的笔记内容，此处补充更完整的量化对比和 3D 制造工艺细节。

HBM 是一种基于 **3D 芯片堆叠技术** 的高阶内存架构。传统 GDDR 显存像"摊煎饼"——内存颗粒平铺在 PCB 板上。HBM 像"盖摩天大楼"——多层 DRAM 芯片垂直堆叠，通过 **硅通孔 (TSV, Through-Silicon Via)** 穿透所有芯片垂直连接，放在与 GPU 核心极近的硅中介层（Silicon Interposer）上。

### HBM 解决了三个致命痛点

#### 1. 打破"内存墙"（Memory Wall）

- **痛点**：GPU 算力增长远超内存带宽增长
- **解决**：GDDR 位宽 32-64 位，HBM 每堆栈位宽 **1024 位**（HBM3e），HBM4 翻倍到 **2048 位**
- **等效带宽**：一条 64 位的 GDDR6 要跑到 ~500 GB/s 需要极高的时钟频率（功耗爆炸）。HBM 靠"加车道"实现同样带宽，频率低得多 → 功耗低得多

#### 2. 突破功耗极限

- **痛点**：传统显存提频率换带宽 → 功耗和发热指数上升
- **解决**：HBM 不拼频率，拼并行度。低频低压 + 万千车道同时跑。**每传输 1GB 数据的功耗远低于 GDDR**（约为后者的 1/3）

#### 3. 压缩物理空间

- **痛点**：平铺的 GDDR 颗粒占据大量 PCB 面积，限制了 GPU 能放的内存总量
- **解决**：3D 堆叠让 HBM 占用的水平空间极小。多个 HBM 堆栈紧贴 GPU 核心。相比 GDDR 节省 **90% 以上**面积，缩短物理距离 → 降低延迟

### HBM 的 3D 制造工艺

- **TSV（硅通孔）**：在 DRAM 芯片上打微米级垂直孔（直径 ~5-10μm），填入铜。数据不绕芯片边缘，直接"坐电梯"贯穿整栋 DRAM 楼层。这是整个 HBM 的灵魂技术。
- **Micro-bumps（微凸块）**：层与层之间、HBM 堆栈与底部逻辑芯片之间，通过海量微金属球（间距 ~40-55μm）精确焊接导通。每个 HBM3 堆栈可能有数千个微凸块。
- **Base Die（底座逻辑芯片）**：堆栈最底层是一块逻辑芯片。负责：① 统筹上面所有 DRAM 层的读写请求 ② 作为 PHY 接口与 GPU 侧的内存控制器高速通信 ③ 管理刷新、纠错（ECC）
- **Silicon Interposer（硅中介层）**：HBM 堆栈和 GPU 核心**共同坐在一块更大的硅基座上**。它们之间通过硅中介层上的微米级导线连接（不是 PCB 上的铜线），信号完整性和速度远超传统 PCB 走线。

### 各代 GPU HBM 规格对比（最完整版）

| | V100 | A100 | H100 | H200 | B200 |
|---|---|---|---|---|---|
| **架构** | Volta | Ampere | Hopper | Hopper Refresh | Blackwell |
| **年份** | 2017 | 2020 | 2022 | 2024 | 2025 |
| **HBM 类型** | HBM2 | HBM2e | HBM3 | HBM3e | HBM3e |
| **总容量** | 16/32 GB | 40/80 GB | 80 GB | 141 GB | 192 GB |
| **带宽** | 900 GB/s | 2.0 TB/s | 3.35 TB/s | 4.8 TB/s | **8.0 TB/s** |
| **堆栈数** | 4 | 5 | 6 | 6 | 8（4/Die） |
| **位宽/Stack** | 1024-bit | 1024-bit | 1024-bit | 1024-bit | 1024-bit |
| **总位宽** | 4096-bit | 5120-bit | 5120-bit | 5120-bit | 2×4096-bit |
| **堆叠层数** | 4 层 | 8 层 | 8 层 | 8 层 → 12 层 | 12 层 |
| **数据速率** | 2.0 Gbps | 3.2 Gbps | 5.2 Gbps | 7.5 Gbps | ~8 Gbps |

### 消费级 vs 数据中心 GPU 的存储对比

| | RTX 4060 Laptop | RTX 4090 | A100 80GB | H100 |
|---|---|---|---|---|
| **显存类型** | GDDR6 | GDDR6X | HBM2e | HBM3 |
| **容量** | 8 GB | 24 GB | 80 GB | 80 GB |
| **带宽** | 272 GB/s | 1008 GB/s | 2039 GB/s | 3350 GB/s |
| **位宽** | 128-bit | 384-bit | 5120-bit | 5120-bit |
| **价格** | ~$1000（整机） | ~$1600 | ~$15000 | ~$30000 |

> **关键洞察**：消费级 GPU 使用 GDDR（平铺颗粒，便宜），数据中心 GPU 使用 HBM（3D 堆叠，贵 10-20×）。你的 RTX 4060 带宽 272 GB/s，A100 是 2039 GB/s——差了 7.5 倍，这决定了两者能跑的 batch size 和模型规模完全不同。**但存储层次的金字塔结构是一样的。**

> A100 的 HBM 带宽 2TB/s 仍然远远慢于 L2 的 4TB/s、SRAM 的 19TB/s。**内存墙不是被打破了，只是被 HBM 推远了一点，核心矛盾仍然存在。**

---

## 六、设计权衡：为什么不能全用快的？

一个自然的疑问：既然 SRAM 这么快，为什么不多放点？把 80GB HBM 全换成 SRAM 不就好了？

### 物理约束 1：芯片面积（最硬性的约束）

**SRAM 的一个 bit 需要 6 个晶体管**（6T SRAM cell，用交叉耦合的反相器对来锁存数据）。**DRAM 只需要 1 个晶体管 + 1 个电容**（1T1C cell，电容存电荷，晶体管控制读写）。

| 存储类型 | 晶体管/bit | 每 bit 面积（7nm） | 1GB 所需面积 |
|---------|-----------|------------------|------------|
| SRAM（6T cell） | 6 | ~0.026 μm² | ~200 mm² |
| DRAM（1T1C cell） | 1 | ~0.004 μm² | ~30 mm² |

**每 bit SRAM 面积是 DRAM 的 5-7 倍。**

A100 芯片总面积 **826 mm²**（TSMC 7nm 工艺）。如果把 80GB HBM 全换成片上 SRAM：
- 80GB × 8 bits × ~0.026 μm² ≈ **16,640 mm²**
- 一块 12 英寸晶圆只能切出 3-4 颗这样的芯片
- 当前最先进的 EUV 光刻机（ASML NXE:3400C）一台 2 亿美元，良率也上不去
- **成本是天文数字，物理上不可行**

### 物理约束 2：功耗

SRAM 比 DRAM 快，但**漏电流（Leakage Current）更大**。6 个晶体管即便不读写也在漏电。

- 80GB SRAM 的静态功耗（什么都不干时）就**超过几千瓦**
- 再加动态功耗（读写时），散热根本解决不了
- H100 整卡功耗 700W，如果用全 SRAM 方案估计轻松突破 5000W——需要的散热系统比显卡本身大 10 倍

### 物理约束 3：光速限制

电信号在硅中的传播速度约 **0.5-1 c**（c = 光速），在 2 GHz 时钟下 1 个 cycle ≈ 0.5ns，信号最多跑 **~1-2 cm**。

- SRAM 必须在计算核心旁（< 1cm），否则信号到不了 → 总面积受限
- HBM 可以离计算核心稍远（硅中介层上相邻，~几 mm）→ 可以做更大
- 如果所有内存都在芯片外部（GDDR），距离更远（PCB 走线，~5-10cm）→ 延迟更大

### 结论

> **多级存储不是一种设计选择，是面积、功耗、光速三条物理定律共同逼出来的唯一解。** SRAM 管速度，DRAM 管容量，各司其职。每一代 GPU 都在同样的约束下微调比例——SRAM 大 30%、L2 大 25%、HBM 带宽 +50%——但金字塔的**形状**不会变。

---

## 七、代际演进：Ampere → Hopper → Blackwell

### 完整规格表

| | A100 (Ampere) | H100 (Hopper) | H200 (Hopper+) | B200 (Blackwell) |
|---|---|---|---|---|
| **年份** | 2020 | 2022 | 2024 | 2025 |
| **制造工艺** | TSMC 7nm N7 | TSMC 4N (5nm) | TSMC 4N | TSMC 4NP |
| **芯片面积** | 826 mm² | 814 mm² | 814 mm² | 2 × ~800 mm²（双 Die） |
| **晶体管** | 54.2B | 80B | 80B | 208B（双 Die） |
| **SM 数量** | 108 | 132 | 132 | 2 × ？ |
| **SRAM/SM** | 192 KB | 256 KB | 256 KB | ~256 KB+ |
| **L2 Cache** | 40 MB | 50 MB | 50 MB | ~2×（双 Die） |
| **HBM 类型** | HBM2e | HBM3 | HBM3e | HBM3e |
| **HBM 容量** | 40/80 GB | 80 GB | 141 GB | 192 GB |
| **HBM 带宽** | 2.0 TB/s | 3.35 TB/s | 4.8 TB/s | **8.0 TB/s** |
| **TDP** | 400W | 700W | 700W | 1000W |
| **FP16 算力** | 312 TFLOPS | 989 TFLOPS | 989 TFLOPS | 2250 TFLOPS |
| **算力/带宽 比** | 156 FLOPS/Byte | 295 FLOPS/Byte | 206 FLOPS/Byte | 281 FLOPS/Byte |

### 三条趋势线

**1. SRAM 缓慢增长（每代 +25-33%）**

```
V100: 128 KB/SM → A100: 192 KB/SM → H100: 256 KB/SM
```

受芯片面积上限约束，不可能暴涨。每代只加一点点。

**2. HBM 带宽快速增长（每代 +50-70%）**

```
A100: 2.0 TB/s → H100: 3.35 TB/s → H200: 4.8 TB/s → B200: 8.0 TB/s
```

通过增加堆叠层数（8→12 层）和提升数据速率（3.2→8 Gbps）实现。

**3. 模型规模增长更快**

```
GPT-3 (2020): 175B 参数 → GPT-4 (2023): ~1.7T → 未来: 10T+
HBM 容量: 80GB → 141 GB → 192 GB → ?
```

**模型参数的增长速度超过了 HBM 容量的增长速度。** 这意味着：无论硬件怎么涨，模型总是先 OOM 的那个。软硬件协同优化只会越来越重要。

### B200 的双 Die 设计

Blackwell B200 把两个 GPU Die 封装在一起，通过 **NVLink-C2C（~900 GB/s）** 互联：

- 每个 Die 有自己的 SM、L2、HBM 控制器
- 总共 192 GB HBM3e（每个 Die 4 个 HBM stack）
- 8 TB/s 总带宽
- 编程上像一个"虚拟大 GPU"（通过 NVLink 做 cache-coherent 内存共享）

> **核心判断**：未来 5-10 年，GPU 存储层次的金字塔结构不会变。硬件侧靠堆叠更多层 HBM 和 bigger die，算法侧靠 FlashAttention 这类 IO-Aware 重写。**两者都很重要，但算法侧可挖掘的空间大得多——因为硬件的"量的增长"赶不上模型规模的"质的膨胀"。**

---

## 八、与 Week 1 的关联：FlashAttention 的硬件解释

Week 1 的 FlashAttention 代码，每一行都可以用 Week 2 的硬件知识重新解读：

| Week 1 代码 | 硬件位置 | 具体物理过程 |
|------------|---------|------------|
| `Qi = Q[i_start:i_end]` | HBM → L2 → L1 → SRAM | 从 HBM 读取一个 Q 块。如果 L2 命中 → 200 cycles；Miss → 600 cycles |
| `Kj = K[j_start:j_end]` | HBM → SRAM | K 块同理。如果 PagedAttention 的 KV Block 常驻 L2，命中率提升 |
| `S = Qi @ Kj.T` | **SRAM + 寄存器** | Tensor Core 执行 MMA（矩阵乘加），Q_i 和 K_j 都在 SRAM 中，结果 S 暂存寄存器 |
| `m_new = max(m, ...)` | **寄存器** | Online Softmax 的状态更新，全程在寄存器，0 cycle 延迟 |
| `O_block * scale + P@V` | **寄存器 + SRAM** | 缩放 + 累加，中间值在寄存器，最终 O_block 在 SRAM |
| 注释"阅后即焚" | SRAM 被覆盖 | 下一次 `j_start` 循环的 `K_j, V_j` 直接覆盖同一块 SRAM 空间 |
| `O[i_start:i_end] = O_block/l` | SRAM → HBM（唯一写回） | 遍历完所有 K,V 块后，唯一一次将结果写入 HBM |

### block_size 为什么是 64？—— SRAM 容量约束的精确计算

一个 Q 块在 A100 的 Shared Memory（~164 KB 可用）中所占的空间：

```
(假设 d=64, block_size=64, FP16 精度)

Q 块:  [64, 64] × 2 bytes = 8 KB
K 块:  [64, 64] × 2 bytes = 8 KB
V 块:  [64, 64] × 2 bytes = 8 KB
S 矩阵: [64, 64] × 2 bytes = 8 KB   (Q @ K^T 的结果，临时)
状态:  m[64] + l[64] + O[64,64] ≈ 0.5 KB
────────────────────────────────
合计:                         ≈ 33 KB  < 128 KB ✓ (安全)
```

block_size = 128 时：
```
Q 块:  [128, 64] × 2 = 16 KB
K 块:  [128, 64] × 2 = 16 KB
V 块:  [128, 64] × 2 = 16 KB
S 矩阵: [128,128] × 2 = 32 KB
────────────────────────────────
合计:                     ≈ 80 KB  > 128 KB (RTX 4060 刚好放不下)
                                  < 164 KB (A100 可以)
```

block_size = 256 时：
```
S 矩阵: [256,256] × 2 = 128 KB  ← 光这一个就快爆了！
合计: > 224 KB  →  任何 GPU 都放不下！
```

> **这就是论文 IO 复杂度公式 O(N²d²/M) 中 M 的物理含义：M 就是 Shared Memory 的大小。M 越大 → block_size 能设得越大 → HBM 访问次数越少 → 算子越快。每一代 GPU 的 SRAM 增大 30%，FlashAttention 天然就能快 10-20%。**

---

## 九、总结：四个"一"

### 一个核心矛盾
**算力增长远超带宽增长。** 从 A100 到 B200，FLOPS 涨了 7 倍，HBM 带宽涨了 4 倍。这个差距永远不会消失——因为物理定律决定了 SRAM 不能太大、HBM 不能太快。

### 一个不变的框架
**四层金字塔（寄存器 → SRAM → L2 → HBM）** 是 GPU 存储层次的不变结构。每层有明确的职责：寄存器管单线程、SRAM 管 Block 内共享、L2 管跨 SM 复用、HBM 管全模型存储。

### 一个核心技能
**用手动管理的 SRAM（`__shared__`）替代自动管理的 L2/HBM。** 这是 AI Infra 工程师优化算子的核心思路。FlashAttention 就是把 Attention 的中间计算从依赖 HBM 变成在 SRAM 中"阅后即焚"。

### 一个未来趋势
**硬件侧微调比例，算法侧重写 IO 模式。** 前者每代给 30%，后者一次给 10-100×。AI Infra 这个方向的价值就在这里——不是等硬件变快，而是让同一块硬件上的软件跑得更聪明。

---

## 十、学习路线

### 阶段 1：理解各层（2 天）

| 天数 | 内容 | 输出 |
|------|------|------|
| 1 | GPU 存储层次全景。画出 A100 和你的 RTX 4060 的两张金字塔图，标注各层的容量/延迟/带宽。对比两者的差异 | 两张金字塔图，贴在笔记里 |
| 2 | 每层的物理本质：6T SRAM cell（寄存器/SRAM）vs 1T1C DRAM cell（HBM）。理解"为什么大就意味着慢" | 能用一句话解释 |

### 阶段 2：深入 Shared Memory（1 天）

| 天数 | 内容 | 输出 |
|------|------|------|
| 3 | Bank Conflict：手算几个例子（stride=1/2/32 分别命中哪些 Bank）。理解 `[32][32+1]` Padding 技巧 | 一张 Bank 映射表（行号→Word 地址→Bank 编号）|

### 阶段 3：结合 Week 1（1 天）

| 天数 | 内容 | 输出 |
|------|------|------|
| 4 | 用本章的知识重新注释 Week 1 的 `flash_attention()` 函数。计算 block_size=64 的 SRAM 预算（哪个数据占多少 KB）。理解为什么 block_size 不能是 256 | 注释版代码 + SRAM 预算表 |

---

## 十一、学习材料

| 优先级 | 材料 | 重点读什么 |
|-------|------|-----------|
| ★★★ | [NVIDIA A100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) | SM 架构图、存储层次表格、L2 Residency Control |
| ★★★ | [NVIDIA H100 Tensor Core GPU Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) | H100 vs A100 的 SM 变化、HBM3 升级 |
| ★★★ | [CUDA C++ Programming Guide — Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory) | `__shared__` 声明方式、Bank Conflict 详解 |
| ★★ | [CUDA C++ Programming Guide — Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy) | 各层内存的访问模式和延迟数量级 |
| ★★ | [GPU Memory Hierarchy: HBM, SRAM, Registers Explained](https://zeroentropy.dev/concepts/gpu-memory-hierarchy/) | 独立博客的通俗解释 |
| ★ | [Cornell Virtual Workshop — GPU Performance Topics (Bank Conflicts)](https://cvw.cac.cornell.edu/cuda-intro/gpu-performance-topics/banks) | Bank Conflict 的交互式教程 |
| ★ | FlashAttention 论文 (Dao et al., 2022) — Section 2.1-2.2 | IO 复杂度公式 O(N²d²/M) 的证明，M=SRAM 容量 |
| ★ | 计算机体系结构：量化研究方法 (Hennessy & Patterson) — 第 2 章 Memory Hierarchy | 存储层次的设计原理（CPU/GPU 通用） |

---

## 十二、个人思考与理解

### GPU 存储层次的本质

GPU 的存储层次**不是一种设计偏好，而是物理定律的必然结果**。快和大在硅基芯片上无法兼得——SRAM 的 6T cell 占 5-7× 面积但快得飞起，DRAM 的 1T1C cell 省面积但需要几百个周期才能访问到。只要还在用硅（而不是光子芯片或超导材料），这个矛盾就解不开。

### 这意味着什么

1. **硬件侧**：每一代 GPU 都在这四层之间重新分配面积和功耗预算。HBM4 带宽翻倍、SRAM 大 30%、L2 更聪明。但金字塔结构不会变——因为存储介质的物理差异是不变的。

2. **软件侧**：**这才是 AI Infra 最活跃的战场。** 硬件的代际改进是线性的（每代 +30-50%），算法的改进可以是数量级的——FlashAttention 证明了一个简单的算法重排就能节省 10-100× 的 HBM 流量。

3. **趋势**：未来会有更多"SRAM-Aware"的算法被重新设计。**懂体系结构的算法工程师比懂算法的硬件工程师更稀缺**——因为软件层面的可优化空间远大于硬件层面。

### 我选择 AI Infra 方向的原因

AI Infra 夹在算法和硬件之间。不是"调参侠"也不是"搬砖工"，而是**理解硬件约束后重新设计计算模式的人**。FlashAttention 就是标志性案例：没有发明新的 Attention 机制，只是换了一种数据搬运方式，就在同一块 A100 上让长上下文训练从"需要 128 张卡"变成"一张卡就够"。

> **这就是我想做的事：找到下一个 FlashAttention。不是等硬件变快，而是让软件跑得更聪明。**
