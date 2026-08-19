# Week 4 学习准备：CODA 论文前置知识清单

> **论文**: CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs
> **arXiv**: https://arxiv.org/abs/2605.19269
> **代码**: https://github.com/HanGuo97/coda-kernels
> **学习周期**: 6 天

---

## 已有基础自查（Week 1-3 回顾）

在开始之前，确认以下概念你都能口述清楚。如果某个点说不清楚，先回顾对应的 Week N 笔记。

| 自查项                                                   | 对应笔记        | 达标？ |
| ----------------------------------------------------- | ----------- | --- |
| Self-Attention 的 Q/K/V 投影 + Softmax + ×V 全流程          | Week 1      | ☐   |
| Transformer Block 结构：Attention → 残差 → Norm → FFN → 残差 | Week 1      | ☐   |
| Memory-bound vs Compute-bound 的判断标准（算术强度）             | Week 2 §零   | ☐   |
| GPU 四层存储金字塔 + 每层的延迟/带宽数量级                             | Week 2      | ☐   |
| Shared Memory Tiled GEMM 的完整代码逻辑                      | Week 3 §4   | ☐   |
| 两个 `__syncthreads()` 各保护什么                            | Week 3 §4.4 | ☐   |
| 协作加载（Cooperative Loading）为什么比单线程循环快                   | Week 3 §4.5 | ☐   |
| Roofline 模型：你的 RTX 4060 上 arithmetic intensity 阈值是多少  | Week 3 §3.3 | ☐   |
| Online Softmax 的分块归约思想                                | Week 1 §二   | ☐   |

---

## 第一部分：必读材料（Day 1，约 3 小时）

> **目标**: 建立 GEMM Epilogue 的 mental model，理解 CODA 要解决的问题发生在 GEMM 输出的哪个阶段。

### 1.1 CUTLASS Epilogue 概念（必读，~60 min）

这是理解 CODA 的**最关键的单点概念**。你 Week 3 手写的 tiled GEMM kernel 结构：

```
外层循环 (遍历 K 维 tile):
    协作加载 A_tile, B_tile    (Global → Shared, ~400 cycles)
    __syncthreads()
    内层乘加循环               (Shared → Register, ~20 cycles/次)  ← 这是 Main Loop
    __syncthreads()
写回 C → HBM                 ← 这是最朴素的 Epilogue
```

**工业级 GEMM（CUTLASS）把"写回"和"写回前能做的事"抽象成了独立的 Epilogue 阶段**——此时 C 的结果还在寄存器/Shared Memory 里，可以几乎零成本地做 element-wise 操作，然后才写回 HBM。

| #   | 材料                                     | 链接                                                                                           | 重点读什么                                                                                   | 时间     |
| --- | -------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------ |
| 1   | CUTLASS Epilogue Visitor Tree (EVT) 教程 | [Colfax Research](https://research.colfax-intl.com/epilogue_visitor_tree/)                   | 前两节：Epilogue 是什么 + 为什么融合能省带宽                                                            | 30 min |
| 2   | CUTLASS Epilogue 代码结构概览                | [DeepWiki](https://deepwiki.com/NVIDIA/cutlass/5.3-epilogue-fusion-and-activation-functions) | "Linear Combination" 和 "Activation Functions" 两节。理解 `D = alpha * acc + beta * C` 这种融合模式 | 20 min |
| 3   | 论文 CODA Section 2（Background）          | [arXiv](https://arxiv.org/abs/2605.19269)                                                    | 只读 Section 2，带着上面学到的 Epilogue 概念去读                                                      | 10 min |

**检验标准**: 能用一句话解释"GEMM Epilogue 是什么，为什么在 Epilogue 里做 element-wise op 能省 HBM 带宽"。

---

### 1.2 CuTe / CUTLASS DSL 快速入门（必读，~60 min）

CODA 的代码基于 **CuTeDSL**（CUTLASS 的 Python DSL）。你不需要精通，但要能看懂代码的结构——"用 Python 描述 tile 切分和线程映射，然后编译成 CUDA kernel"。

| # | 材料 | 链接 | 重点读什么 | 时间 |
|---|------|------|-----------|------|
| 1 | CuTe 官方 Quickstart | [CUTLASS docs](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md) | 只读前三个例子：Layout → Tensor → Copy。理解 CuTe 的核心抽象 | 20 min |
| 2 | 知乎：CUTLASS CuTe 实战（一） | [知乎](https://zhuanlan.zhihu.com/p/690703999) | 非常详细的中文教程。重点看 VectorAdd 的完整例子（对比 naive CUDA vs CuTe 实现） | 30 min |
| 3 | CUTLASS CuTe 示例代码仓库 | [GitHub](https://github.com/leimao/CUTLASS-Examples) | 浏览目录结构即可，不用跑。了解 CuTe 能做什么 | 10 min |

**检验标准**: 能看懂 CODA 代码仓库里一个 kernel 文件的大致结构：tile 声明 → main loop → epilogue → launch。

---

### 1.3 三个 Transformer 操作的数学形式（速查，~45 min）

CODA 的核心贡献是把下面这些操作塞进 GEMM Epilogue。你需要知道它们**数学上到底在算什么**，才能理解为什么可以"重写为 epilogue 操作"。

#### A. RMSNorm（Root Mean Square Layer Normalization）

现代 LLM（LLaMA、Mistral、Qwen）不用 LayerNorm，用 RMSNorm——去掉了"减均值"这步，更快。

$$y = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}$$

**计算拆解**：
1. **Reduction 阶段**：算 $\sum x_i^2$（tile reduction，和 Week 1 的 online softmax 归约思想一样）
2. **Element-wise 阶段**：除以 RMS，乘以 $\gamma$（纯 element-wise，在寄存器里完成）

**对应 CODA 原语**：Tile Reduction + Element-wise Transform

> 🔗 NVIDIA Transformer Engine RMSNorm 实现: [GitHub](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/ops/basic/rmsnorm.py)

#### B. SwiGLU（Gated Activation）

$$\text{SwiGLU}(x) = \text{SiLU}(xW_{\text{gate}}) \odot (xW_{\text{up}})$$

- 两个并行的 GEMM：`xW_gate` 和 `xW_up`
- SiLU = $x \cdot \sigma(x)$（sigmoid 门控，纯 element-wise）
- $\odot$ = element-wise 乘法（Hadamard product）

**对应 CODA 原语**：两个 GEMM + Element-wise Transform（SiLU + multiply）

> 🔗 SwiGLU 论文: [Shazeer 2020, GLU Variants](https://arxiv.org/abs/2002.05202) — 只需看 Section 2 的公式和表格

#### C. RoPE（Rotary Position Embedding）

对 Q 和 K 的每对相邻维度做 2D 旋转：

$$\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos\theta_i & -\sin\theta_i \\ \sin\theta_i & \cos\theta_i \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

- 每个位置有预计算的 $\cos\theta$, $\sin\theta$
- 纯 element-wise 操作：不改变 shape，不涉及跨元素归约

**对应 CODA 原语**：Element-wise Transform

> 🔗 RoPE 论文: [Su et al., RoFormer](https://arxiv.org/abs/2104.09864) — 只需看 Section 3.1 的旋转公式

| # | 任务 | 时间 |
|---|------|------|
| 1 | 在纸上写出 RMSNorm 的完整数学形式，标注哪步是 reduction、哪步是 element-wise | 10 min |
| 2 | 在纸上画出 SwiGLU 的计算图：两个 GEMM → SiLU → multiply | 10 min |
| 3 | 理解 RoPE 旋转矩阵只作用在相邻维度对上，不改变 shape | 10 min |
| 4 | 思考：如果你在 Week 3 的 `sgemm_tiled` kernel 的写回循环里加一行 `acc = acc * 2.0f`，这算不算是 epilogue 操作？（答案：算！这就是最简单的 epilogue element-wise transform） | 5 min |

---

## 第二部分：论文阅读计划（Day 2-4，约 7 小时）

### Day 2：Motivation & Background（~2h）

| 步骤 | 内容 | 时间 |
|------|------|------|
| 1 | 读 Section 1（Introduction）。**核心问题**：随着 FP8/FP4 让 GEMM 越来越快，memory-bound 操作（norm、残差、激活函数）的相对开销为何反而上升？ | 30 min |
| 2 | 读 Section 2（Background & Related Work），现在你有了 Day 1 的 Epilogue 知识再看这段会很顺 | 30 min |
| 3 | 读 Section 3 的开头部分（五类 Epilogue 原语的总体介绍）。对照下表理解每类原语解决什么问题 | 30 min |
| 4 | 用你自己的话写一段总结：CODA 要解决什么问题？现有的解决方案（手写 CUDA kernel fusion）为什么不够好？ | 30 min |

**五类 Epilogue 原语速查表**：

| 原语 | 解决的问题 | 你的 Week 1-3 知识锚点 |
|------|-----------|----------------------|
| **Element-wise Transform** | 残差加法、激活函数（SiLU/GELU）、RoPE 旋转 | Week 3 写回循环里加一行 `acc = f(acc)` 就是最简单的形式 |
| **Vector Load/Store** | 广播 RMSNorm 的 gamma 权重向量 | Shared memory 上一根向量被整个 block 的线程读（类似 Week 3 的 As 广播） |
| **Matrix Tile Load/Store** | 保存中间激活供反向传播用 | Week 3 的协作加载 → 反过来就是协作存储 |
| **Tile Reduction** | 局部 RMS 的平方和、block log-sum-exp | **Week 1 的 online softmax 就是 tile reduction！** 你已经懂了 |
| **Stateful Transform** | 在线归一化的 running max/sum-exp | **Week 1 的 FlashAttention online softmax 状态更新也是这个！** |

> 你会发现：**五类原语里有两类（Tile Reduction 和 Stateful Transform）在 Week 1 学 FlashAttention 时已经见过了**，只是当时不叫这个名字。

---

### Day 3：核心机制——代数重参数化（~3h）

这是 CODA 论文最精彩的部分，也是最有学习价值的地方。

| 步骤 | 内容 | 时间 |
|------|------|------|
| 1 | 读 Section 3 剩余部分。理解 CODA 是怎么把 GEMM-Residual-RMSNorm 重写为单个 GEMM-with-epilogue | 45 min |
| 2 | 读 Section 4（前向传播的重写方法）。重点关注：GEMM-RMSNorm-GEMM、GEMM-Residual-PartialRMS-GEMM、SwiGLU 的重写 | 45 min |
| 3 | 在纸上画出 "Naive 执行" vs "CODA 执行" 的对比数据流图（参照下面的模板） | 30 min |
| 4 | 读 Section 5（反向传播）。和 Week 1 的 FlashAttention 反向重计算做对比——异同在哪里？ | 30 min |
| 5 | 读 Section 6（实验结果）。重点关注：为什么 batch size 越小 CODA 收益越大？（提示：memory-bound 程度） | 30 min |

**"Naive vs CODA" 数据流对比模板**（拿 GEMM-Residual-RMSNorm 举例）：

```
Naive 执行（3 个 kernel，6 次 HBM 跨越）:
┌──────┐    ┌──────────┐    ┌──────┐    ┌──────────┐    ┌──────┐
│ GEMM │───→│ HBM 写回  │───→│ 读回  │───→│ HBM 写回  │───→│ 读回  │
│kernel│    │   C_mid   │    │ C_mid│    │  C_norm   │    │C_norm │
└──────┘    └──────────┘    └──────┘    └──────────┘    └──────┘
                               │                            │
                        残差相加 + RMSNorm              下一层 GEMM
                        (memory-bound!)              (需要读 C_norm)

CODA 执行（1 个 kernel，1 次 HBM 写出）:
┌──────────────────────────────────────────────┐
│ GEMM Main Loop                               │
│   → 累加结果在寄存器                          │
│   → Epilogue: +residual → reduction(RMS)     │
│     → normalize → *gamma                     │
│   → 最终结果一次写回 HBM                       │
└──────────────────────────────────────────────┘
```

---

### Day 4：代码实战（~2h）

| 步骤 | 内容 | 时间 |
|------|------|------|
| 1 | `git clone https://github.com/HanGuo97/coda-kernels` | 5 min |
| 2 | 读 README，按照安装步骤配环境（pip install CuTeDSL 等） | 30 min |
| 3 | 跑通最简单的 example（比如 GEMM + element-wise op fusion） | 30 min |
| 4 | 打开一个 kernel 的 Python DSL 源码，对照论文的 Section 3/4，找到：① tile 声明、② main loop、③ epilogue 操作、④ launch config | 30 min |
| 5 | 尝试改一个参数（比如 tile size），看看性能变化 | 15 min |
| 6 | 记录你看到的代码结构和论文描述的对应关系 | 10 min |

---

## 第三部分：深入研究（Day 5-6，约 4 小时）

### Day 5：回顾 + 整理笔记（~2h）

| 任务 | 输出 |
|------|------|
| 用你自己的话写 CODA 的 one-pager：问题 → 方法 → 关键 insight → 结果 | 一篇 ~500 字的总结，放在 Week 4 笔记里 |
| 画出 CODA 的五类 Epilogue 原语与 Transformer 操作的映射表 | 一张表 |
| 思考：CODA 的方法有没有局限？什么情况下 epilogue fusion 不 work？ | 一段分析（提示：跨 token 的依赖操作，如 Attention 的 Q×K^T，无法在单 GEMM 的 epilogue 里完成） |
| 思考：CODA 的思想能不能用到其他领域（比如推荐系统、图神经网络）？ | 一段分析 |

### Day 6：扩展阅读（可选，~2h）

| # | 材料 | 为什么值得读 |
|---|------|------------|
| 1 | [FlashAttention-3 论文](https://arxiv.org/abs/2407.08608) | 和 CODA 互补——FA3 优化 Attention 内部的 IO，CODA 优化 Attention 前后的 norm/residual/activation 的 IO |
| 2 | CUTLASS 官方 GEMM 教程 | [Efficient GEMM in CUTLASS](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)，从 main loop → epilogue 的完整流程，巩固 Week 3 + Week 4 的知识链路 |
| 3 | [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) | Stanford HazyResearch 的 GPU kernel DSL，和 CuTeDSL 类似的思路但更轻量，可以对比理解"DSL 如何抽象 epilogue" |
| 4 | [TileLang](https://github.com/facebookresearch/tilelang) | Meta 的 tile-level DSL，CODA 的作者之一也参与了这个项目。理解 tile-level 编程的生态 |

---

## 第四部分：自检清单（学完后逐条过）

- [ ] 能用一句话解释 GEMM Epilogue 是什么
- [ ] 能说出 CODA 五类 Epilogue 原语各对应什么 Transformer 操作
- [ ] 能画出 GEMM-Residual-RMSNorm 在 Naive 和 CODA 两种方式下的数据流对比图
- [ ] 能解释为什么 batch size 越小 CODA 收益越大
- [ ] 能解释 CODA 反向传播加速 1.6-1.8× 的原理
- [ ] 能说出 CODA 为什么只支持单 GPU（当前限制）
- [ ] 能在 CODA 代码仓库里找到 epilogue 定义的位置
- [ ] 能解释 CODA 和 FlashAttention 各自解决什么问题，为什么互补

---

## 附：关键资源汇总

### 论文 & 代码

| 资源 | 链接 |
|------|------|
| CODA 论文 | https://arxiv.org/abs/2605.19269 |
| CODA 代码 | https://github.com/HanGuo97/coda-kernels |
| CUTLASS 仓库 | https://github.com/NVIDIA/cutlass |
| CuTe 入门文档 | https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md |
| CuTe 知乎实战 | https://zhuanlan.zhihu.com/p/690703999 |
| CuTe 示例代码 | https://github.com/leimao/CUTLASS-Examples |

### Epilogue 概念

| 资源 | 链接 |
|------|------|
| CUTLASS Epilogue Visitor Tree 教程 | https://research.colfax-intl.com/epilogue_visitor_tree/ |
| CUTLASS Epilogue Fusion 文档 | https://deepwiki.com/NVIDIA/cutlass/5.3-epilogue-fusion-and-activation-functions |
| Efficient GEMM in CUTLASS | https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md |

### Transformer 操作

| 操作 | 资源 | 读什么 |
|------|------|--------|
| RMSNorm | [NVIDIA Transformer Engine 实现](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/ops/basic/rmsnorm.py) | 前向传播的公式 + 代码结构 |
| SwiGLU | [GLU Variants 论文](https://arxiv.org/abs/2002.05202) | Section 2 公式 |
| RoPE | [RoFormer 论文](https://arxiv.org/abs/2104.09864) | Section 3.1 旋转公式 |

### 扩展阅读

| 资源 | 链接 |
|------|------|
| FlashAttention-3 | https://arxiv.org/abs/2407.08608 |
| ThunderKittens DSL | https://github.com/HazyResearch/ThunderKittens |
| TileLang | https://github.com/facebookresearch/tilelang |

---

## 学习日历（建议）

| 天 | 上午（~1.5h） | 下午（~1.5h） |
|----|-------------|-------------|
| **Day 1** | 1.1 CUTLASS Epilogue 概念（60 min） | 1.2 CuTe 快速入门 + 1.3 三个 Transformer 操作（90 min） |
| **Day 2** | CODA Section 1-2（90 min） | CODA Section 3 开头 + 五类原语总结（90 min） |
| **Day 3** | CODA Section 3 剩余 + Section 4 前向重写（90 min） | CODA Section 5 反向 + Section 6 实验（90 min） |
| **Day 4** | 代码实战：clone + 跑通 + 读源码（90 min） | 改参数实验 + 写代码-论文对照笔记（90 min） |
| **Day 5** | 回顾 + 写 one-pager 总结（60 min） | 思考局限 + 扩展 + 自检清单（60 min） |
| **Day 6** | 扩展阅读（可选） | 整理笔记，准备 Week 5 |
