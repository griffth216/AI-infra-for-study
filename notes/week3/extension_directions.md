# Week 3 扩展方向

> Week 3 笔记已建立 block-level tiling 的完整理解。以下四条路线按依赖关系排列，每一条都标注了前置需求、具体可做事项、入门材料和时间估算。

---

## 路线 A：Warp-level Tiling + Register Tiling（⭐ 最优先）

**前置需求**：Week 3 的 shared memory tiled GEMM 已跑通并理解

**为什么值得走**：当前 kernel 内层循环 `for (int k = 0; k < TILE; k++)` 每次都从 Shared Memory 读 `As[threadIdx.y][k]` 和 `Bs[k][threadIdx.x]`——延迟 ~20 cycles/次。如果先把数据搬到寄存器（~1 cycle），内层循环就完全脱离 Shared Memory 了。理论加速：再快 ~20 倍（叠加在 block tiling 已有的提速之上）。这步做完后，你的 kernel 离 cuBLAS 的性能差距会缩小到 2-3 倍以内。

**具体可做事项**：

1. 在 block tile（32×32）内部，把它分成 4 个 warp tile（每个 warp 负责 16×16 的一个子块）
2. 每个线程从 Shared Memory 用 `float4`（128-bit）向量化 load 数据到寄存器
3. 在寄存器上完成内层乘加循环（不再访问 Shared Memory）
4. 用 ncu 对比 block-only vs block+warp tiling 的 `l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum`（Shared Memory Load 事务数）

**入门材料**：
- [siboehm: CUDA Matrix Multiplication](https://siboehm.com/articles/22/CUDA-MMM) Step 4-6（1D/2D Blocktiling + Vectorized Loads）
- PMPP 第 6 章 "Performance Considerations"

**时间估算**：2-3 天

---

## 路线 B：Tensor Core 编程（从 CUDA Core 到专用硬件）

**前置需求**：路线 A 完成后效果更好（理解了寄存器编排后再用 Tensor Core 是水到渠成）

**为什么值得走**：当前代码跑在 CUDA Core 上（FP32，RTX 4060 峰值 ~15 TFLOPS）。Tensor Core 是 Ada Lovelace 架构中专门加速矩阵乘法的硬件单元，FP16 模式下能提供 **~60 TFLOPS（dense）到 ~120 TFLOPS（sparse）**——你目前连这张卡 1/4 的算力都还没用上。AI Infra 方向上所有推理/训练框架最终都落到 Tensor Core 上。

**具体可做事项**：

1. 用 `nvidia-smi` 和 `cudaDeviceGetAttribute` 查询你的 RTX 4060 上 Tensor Core 的具体数量和规格
2. 用 CUDA `wmma`（Warp Matrix Multiply-Accumulate）API 替换内层循环——只需改 ~20 行代码，kernel 就从 CUDA Core 切到 Tensor Core
3. 写一个 `half` 精度的 HGEMM，理解 FP16 的数值范围（max ~65504）和 loss scaling
4. 对比 `cublasGemmEx`（cuBLAS 的 Tensor Core GEMM）和你手写 WMMA kernel 的 GFLOPS——亲身感受 "手写优化 vs 库" 的差距

**入门材料**：
- [CUDA C++ Programming Guide — Warp Matrix Multiply-Accumulate](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [CUTLASS: Basic GEMM with Tensor Cores](https://github.com/NVIDIA/cutlass/blob/main/examples/01_basic_gemm/basic_gemm_tensor_op.cu)
- [NVIDIA Tensor Core 白皮书](https://www.nvidia.com/en-us/data-center/tensor-cores/)

**时间估算**：2-3 天

---

## 路线 C：Double Buffering + 异步拷贝（计算与访存重叠）

**前置需求**：Week 3 的 shared memory tiled GEMM 已跑通

**为什么值得走**：当前 kernel 的结构是：

```
load → __syncthreads() → compute → __syncthreads() → load next...
  ↑                          ↑
  串行！加载时算力闲着        计算时带宽闲着
```

Double buffering 用**两份** shared memory：

```
Buffer 0: load tile 0 → compute tile 0 ─────────────────────→ load tile 2 → compute tile 2
Buffer 1: ────────────────→ load tile 1 → compute tile 1 ────────────────────────────────
                             ↑
                       加载和计算重叠！
```

加上 RTX 4060（Ada Lovelace）支持的 `cp.async` 指令（真正的异步拷贝，不阻塞 SM），可以在硬件层面实现计算与访存的流水线化。这跟你 Week 1 学的 Prefill/Decode 流水线化是同一个思想。

**具体可做事项**：

1. 把 `__shared__ float As[TILE][TILE]` 改为 `__shared__ float As[2][TILE][TILE]`（双缓冲）
2. 用 `cp.async` 替代普通的 `=` 赋值完成 Global → Shared 的数据搬运
3. 用 `cp.async.commit_group()` 和 `cp.async.wait_group()` 管理异步拷贝的同步
4. 在 ncu 的时间线视图中观察算力单元和 Load/Store 单元的并行利用情况

**入门材料**：
- [CUDA Async Data Copies](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#async-data-copies)
- [CUTLASS Pipeline 文档](https://github.com/NVIDIA/cutlass/blob/main/media/docs/pipeline.md)（解释了三阶段 software pipeline）
- [NVIDIA GTC 2022: Async Data Movement in CUDA](https://developer.nvidia.com/blog/advanced-cuda-programming-goal-oriented-approach/)

**时间估算**：1-2 天

---

## 路线 D：读 CUTLASS 源码（将抽象映射到硬件）

**前置需求**：路线 A + B 完成后效果最佳（有了 warp tiling 和 Tensor Core 的实操经验后再看 CUTLASS 的抽象，不会觉得是一堆看不懂的模板）

**为什么值得走**：Week 3 笔记里提了不下五次 "CUTLASS 的三层 tiling"。CUTLASS 把 ThreadblockTile、WarpTile、ThreadTile 抽象成了 C++ 模板参数。花一个下午对着源码把模板参数 → 实际硬件行为的映射关系搞清楚，之后遇到任何 GEMM 优化的论文或算子，看一遍就能分解出它的 tiling 策略。

**具体可做事项**：

1. 克隆 [CUTLASS](https://github.com/NVIDIA/cutlass)，编译 `examples/01_basic_gemm/basic_gemm.cu`
2. 画出 BlockTile → WarpTile → ThreadTile 的实际尺寸和线程映射：
   - `ThreadblockShape::kM=128` 是什么意思？
   - 实际 launch 了多少线程？
   - 每个 warp 负责 shared memory tile 的哪一块？
   - 每个线程的寄存器里存了 C 的哪几个元素？
3. 在你的 RTX 4060 上用 ncu 跑 CUTLASS kernel，对比它和你的 tiled kernel 的 occupancy、shared memory 用量、achieved GFLOPS

**入门材料**：
- [CUTLASS Quick Start](https://github.com/NVIDIA/cutlass#quick-start)
- [CUTLASS Efficient GEMM 文档 Part 1-3](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)
- [CUTLASS 代码走读视频 (GTC 2020)](https://developer.nvidia.com/gtc/2020/video/s21745)

**时间估算**：2-3 天

---

## 推荐执行顺序与依赖关系

```
路线的依赖关系和时间投入：

  A (Warp Tiling)  ← 最自然的下一步，建议本周就开始试
  │                 （原笔记 §5.3 和新增 §8 有展开内容）
  │
  ├── B (Tensor Core)  ← 做完 A 之后，把 FP32 kernel 改成 FP16 Tensor Core
  │
  ├── C (Double Buffering)  ← 和 B 可以并行学，不冲突
  │
  └── D (CUTLASS 源码)  ← A+B+C 都摸过后再读，否则会被模板绕晕
```

**建议节奏**：每周走一条路线。A 这周可以和 Week 4 的新内容并行推进（写代码只需要半天，剩下的是理解+ncu 分析）。
