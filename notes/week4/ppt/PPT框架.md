# CODA 论文解读 PPT 框架（修订版 v2）

> 严格遵循 `汇报架构.md` · 总 39 页 · 10 页章节过渡 · 图随文走 · 时间对齐页码 · 💡 标记个人思考
>
> **核心改进**：
> - 每节入口增加过渡页（IKB 满屏 + §编号 + 标题 + 一句话定位）
> - 图和论证文字不再分家（同一页）
> - 页码按汇报时间比例分配（~1 页/min）
> - 全文 7 处 💡 标注个人思考

---

## 第 1 页 · 封面

- **版式**：S01 Cover（IKB 满屏）
- **内容**：
  - 主标题：CODA：将 Transformer 块重写为 GEMM-Epilogue 程序
  - 副标题：论文解读 · arXiv: 2605.19269
  - 底部：HanGuo97/coda-kernels · 2026.07.16
- **配图**：无

---

## 第 2 页 · 过渡页 — §1 问题动机

- **版式**：章节过渡页（IKB 满屏 + 大字）
- **内容**：
  - §1
  - 问题动机
  - GEMM 已逼近硬件极限，非 GEMM 是最后的主战场
- **配图**：无

---

## 第 3 页 · §1 — 论文图 1 + 数据解读（图随文走）

- **版式**：左图右文
- **内容**：
  - 左：论文图 1（BF16 vs FP8 饼图对比）
  - 右（三层递进）：
    1. GEMM 逼近极限：cuBLAS/CUTLASS 十余年迭代，FP8 Tensor Core 利用率 80-90%，手工优化空间极小
    2. 阿姆达尔定律：GEMM 47%→33%（被 FP8 加速），Others 17%→28%（暴涨），非 GEMM 总占比 53%→67%
    3. Memory Wall：H100 FP8 = 3958 TFLOPS vs HBM3 = 3.35 TB/s。未来 FP4 只会更严重
  - 💡 "GPU 绝大部分时间没在做矩阵乘法，而是在等显存搬运数据。GEMM 优化已是红海，非 GEMM 融合才是蓝海——这是 CODA 存在的理由。"
- **配图**：`01-fig1-bf16-fp8.png`

---

## 第 4 页 · 过渡页 — §2 CODA 核心思路

- **版式**：章节过渡页
- **内容**：
  - §2
  - CODA 核心思路
  - 不动 GEMM 主循环，只在尾部做文章——"寄生"策略
- **配图**：无

---

## 第 5 页 · §2 — 论文图 2 + 核心公式（图随文走）

- **版式**：上图下文
- **内容**：
  - 上：论文图 2（顶行传统 vs 底行 CODA）
  - 下（左中右三列）：
    - 左 · 传统的问题：粉白相间，每个白块 = 一次 HBM 往返。中间张量频繁进出 HBM，GPU 计算核心在白色方块执行时闲置等数据
    - 中 · CODA 的答案：白块被"吞噬"进粉色 GEMM 块尾部。核心公式 GEMM: h=xW，Epilogue: y[i,j]=f[i,j](h[i,j])。唯一约束：tile-locality
    - 右 · 策略本质：CODA 不碰 GEMM 主循环（CUTLASS 专家十年磨出来的），只在寄存器→HBM 的最后一步"顺手"干掉非 GEMM 操作
- **配图**：`02-fig2-naive-coda.png`

---

## 第 6 页 · §2 — 论文图 3 + 架构阐释（图随文走）

- **版式**：左图右文
- **内容**：
  - 左：论文图 3（Mainloop-Epilogue 结构）
  - 右：
    - Main Loop — 固定不动。分块策略、TMA 流水线、双缓冲、寄存器分配。CODA 绝不乱碰
    - Epilogue — 可编程，CODA 的战场。数据还在寄存器里热乎着
    - HBM Write — 最终结果一次性写回。中间结果从未离开寄存器
- **配图**：`03-fig3-mainloop-epilogue.png`

---

## 第 7 页 · 过渡页 — §3 传统方法的困境

- **版式**：章节过渡页
- **内容**：
  - §3
  - 传统方法的困境
  - 前向融合走得越远，反向传播就越痛苦——为 Theorem 1 做铺垫
- **配图**：无

---

## 第 8 页 · §3 — 前向≠反向 + 手写融合的工程代价

- **版式**：左文右文（双栏对照）
- **内容**：
  - 左栏 · 前向融合 ≠ 反向能用（三个根本原因）：
    - 数学公式变了：前向 SwiGLU f(x)=silu(x_g)⊙x_u → 反向含 sigmoid 导数、指数、乘法链路
    - 前向省掉的中间结果，反向又要回来：要么妥协写回（打破融合初衷），要么重计算
    - 内存对齐全乱了：前向按行读取 → 反向转置后按列读取，合并访存被破坏
  - 右栏 · 手写融合的工程代价：
    - 前向手工写一套 CUDA/Triton → 反向必须原封不动再受一遍苦
    - FlashAttention 前向出来几个月后反向才出来——同一个团队，只是方向反了
  - 底部结论："前向 GEMM + Epilogue"如果不能让反向自动继承，就没有从根本上解决问题
- **配图**：无

---

## 第 9 页 · 过渡页 — §4 五大原语

- **版式**：章节过渡页
- **内容**：
  - §4
  - 五大原语：CODA 的工具箱
  - 只暴露能高效映射到 GPU 硬件的操作——五类原语，覆盖 Transformer 99% 的非 GEMM 计算
- **配图**：无

---

## 第 10 页 · §4 — 五类原语速查表

- **版式**：整页大表
- **内容**：

| 原语 | 做什么 | 硬件位置 | Transformer 对应 |
|------|--------|---------|-----------------|
| Elementwise/Pairwise Map | 逐元素/成对变换 | 寄存器内 | SwiGLU, RoPE, 残差加法 |
| Vector Load/Store | 广播一维向量 | Shared Memory | RMSNorm 的 γ 权重 |
| Tile Load/Store | TMA 异步搬运二维块 | HBM ↔ SM | 残差流、保存激活值 |
| Tile Reduction | 分块内局部规约 | Warp Shuffle | RMS 平方和、Cross Entropy LSE |
| Stateful Transform | 寄存器内状态机 | 寄存器 + 状态变量 | Online Softmax 的 m, l |

- 底部标注：五类原语的精简是刻意为之——只暴露"能高效映射到硬件"的操作。Week 1 学的 Online Softmax 的 running max/sum 就是 Stateful Transform
- **配图**：无（自绘表）

---

## 第 11 页 · 过渡页 — §5 前向传播的代数重构

- **版式**：章节过渡页
- **内容**：
  - §5
  - 前向传播的代数重构
  - 三种模式，统一框架：将非 GEMM 操作全部改写为 GEMM + Epilogue
- **配图**：无

---

## 第 12 页 · §5.1 — 公式 + 传统做法 + 缺陷 + 图 4（图随文走）

- **版式**：左文右图
- **内容**：
  - 左：
    - 公式：y = RMSNorm(xW₀+z, γ)W₁ = (r(xW₀+z)⊙γ)W₁
    - 出现在 3 处：Attention 输出→残差→RMSNorm→MLP 门控 / MLP 降级→残差→RMSNorm→QKV / 最终 MLP→残差→RMSNorm→LM Head
    - 传统致命缺陷：GEMM1→写回→RMSNorm（所有线程等 r 算完）→写回→GEMM2。两个 GEMM 被全局同步硬生生阻断
    - 核心观察：残差加和 ×γ 是 tile-local，但 r 需要在隐藏维度（d=4096~8192）上做全局平方和——这是瓶颈
  - 右：论文图 4（代数重构流程）
- **配图**：`05-fig4-gemm-rmsnorm.png`

---

## 第 13 页 · §5.1 — CODA 三步编排（图随文走）

- **版式**：三列流程 + 底部代码
- **内容**：
  - Step 1 · GEMM1 + Epilogue1：h₀=xW₀ → h₁=h₀+z（残差加，Elementwise Map）→ h₂=h₁⊙γ（RMSNorm 权重缩放，Vector Load）→ partialRMS(h₁²)（Tile Reduction）
  - Step 2 · 微型辅助 kernel：合并 ~32 个局部标量 → 全局 r。输入量从 O(D) 降为 O(num_tiles)
  - Step 3 · GEMM2 + Epilogue2：h₃=h₂W₁（**此时不需要 r，直接开始算！**）→ y[i,j]=r[i]·h₃[i,j]（最后才乘 r）
  - 💡 与 5.2/5.3 的关键区别：这个模式跨越两个 GEMM，不能单次 kernel launch 完成 → 拆成三步编排。5.2/5.3 是单 GEMM 内 compose() 搞定
  - 底部代码——ops.py 三步编排：

```python
def gemm_residual_rmsnorm_gemm_fwd(x, y, w_a, w_b, w_n, ...):
    x_out, s, h = gemm_residual_partial_rmsnorm(A=y, B=w_a, C=x, W=w_n, ...)
    rstd_out = compute_rstd(s=s, eps=eps)
    z_out = gemm_rmsnorm(A=h, B=w_b, R=rstd_out)
    return x_out, y_out, z_out, rstd_out
```

- **配图**：`05-fig4-gemm-rmsnorm.png`（复用）+ `06-fig5-benchmark.png`

---

## 第 14 页 · §5.1 — 代数证明 + 对比表

- **版式**：上中下三段
- **内容**：
  - 上 · 代数证明（数乘结合律）：y=(r(xW₀+z)⊙γ)W₁ = r((xW₀+z)⊙γ)W₁。r 是标量（整行共享），矩阵乘法对标量满足结合律 → r 推迟到 GEMM2 尾部应用
  - 中 · 优化优势对比：

| 维度 | 传统 | CODA |
|------|------|------|
| GEMM 间的同步等待 | 必须等 r 算完 | 零等待，r 延迟到 GEMM2 尾部 |
| RMSNorm 开销 | 独立 kernel，读整行求平方和 | 两级规约：局部算 + 辅助合并 |
| 辅助规约输入量 | O(D)（完整张量） | O(num_tiles)（~32 个标量） |

- **配图**：无（纯文字+表）

---

## 第 15 页 · §5.2 — 公式 + 传统做法 + 缺陷

- **版式**：上图下文
- **内容**：
  - 上 · 公式：h = xW，ha[i,j], hb[i,j] = split(h[i,j])，y[i,j] = f(ha[i,j], hb[i,j])
    - SwiGLU：f(g,u) = silu(g)⊙u（降维）| RoPE：f(x₂ₖ,x₂ₖ₊₁) = 2D 旋转变换（保维）
  - 下 · 传统致命缺陷：
    - GEMM 算出膨胀矩阵 H（SwiGLU 需 2× 宽度）→ 完整写回 HBM（数百 MB）→ 启动独立激活 kernel → 跨半个矩阵拉取 ha[j] 和 hb[j]（物理距离 = D，cache miss 严重）→ 算完再写回
    - 中间膨胀矩阵两次完整 HBM 往返
- **配图**：无（铺垫，图在下一页）

---

## 第 16 页 · §5.2 — CODA 三步闭环 + 图 7（图随文走）

- **版式**：左文右图
- **内容**：
  - 左 · 黄金闭环三步（三个独立视觉卡片）：
    - ① 算法层——权重列交错重排（离线，一次性）：传统 W=[W_a,0,...,W_b,0,...] → CODA W_interleaved=[W_a,0,W_b,0,W_a,1,W_b,1,...]。效果：ha[j] 和 hb[j] 从相隔 D 列→物理相邻（2j 和 2j+1）
    - ② 硬件层——Hopper 累加器布局暴露（全文最关键的一条硬件依赖）：Hopper 架构在 Epilogue 阶段向程序员暴露 Tensor Core 累加器物理分布——相邻输出元素天然、必然落在同一线程寄存器中。**不是所有 GPU 都有此特性**——消费级显卡或不开放累加器布局的架构上，CODA 的零开销优势会被直接击穿
    - ③ 闭环：线程一低头，寄存器里恰好是需要配对的两个数 → 直接应用 f，零显存交通，零跨线程通信
  - 右：论文图 7（配对激活示意）
- **配图**：`07-fig7-pairwise.png`

---

## 第 17 页 · §5.2 — 代数证明 + 对比表 + 代码

- **版式**：上中下三段
- **内容**：
  - 上 · 代数证明：矩阵乘法每列独立 → 列重排只改变输出布局不改变计算结果。每个 (2j,2j+1) 对独立 → tile-local 成立。降维时 Epilogue 自动压缩布局（N→N/2），保维时保持不变
  - 中 · 优化优势对比：

| 维度 | 传统 | CODA |
|------|------|------|
| HBM 写回次数 | 2-3 次 | 1 次 |
| 中间膨胀矩阵 | 完整写回 HBM（2×宽度） | 从未离开寄存器 |
| 配对数据获取 | 跨半个矩阵（cache miss） | 同寄存器内，零延迟 |
| Kernel launch 数 | 2（GEMM + 激活 kernel） | 1（融合） |

  - 下 · 代码：

```python
# registry.py — 三行搭一个 GEMM+SwiGLU
GemmSwiGLU = Gated(fn=gate_fn_map["swiglu"]).bind(name="GemmSwiGLU", gemm_cls=GemmSm90)

# activation.py — Pairwise：tRS_rD[2i] 和 tRS_rD[2i+1] 天然同线程
class Pairwise(Epilogue):
    def visit(self, ...):
        for i in range_constexpr(...):
            tRS_rAuxOut[2*i], tRS_rAuxOut[2*i+1] = self.fn(tRS_rD[2*i], tRS_rD[2*i+1])
```

- **配图**：无（代码页）

---

## 第 18 页 · §5.3 — 公式 + 传统缺陷 + CODA 流水线 + 图 8（图随文走）

- **版式**：左文右图
- **内容**：
  - 左：
    - 公式：h_i = x_i W_lm，ℓ_i = -h_i,y_i + log Σ_k exp(h_i,k)，V ≥ 32K
    - 传统致命缺陷：GEMM 算出 B×L×V 的 Logits（batch=2, seq=4096, V=32000 → FP16 约 500MB）→ 完整写回 HBM → Softmax kernel 重新读出。训练中最大的单次 HBM 写入
    - CODA 流水线：GEMM 分块产出 256 个 logits（寄存器）→ Epilogue 检查 target label + Stateful Transform 维护 running m,l → 只写回几个标量 → 辅助 kernel 合并
  - 右：论文图 8（核函数级加速比，Cross-Entropy 列）
- **配图**：`08-fig8-kernel-speedup.png`

---

## 第 19 页 · §5.3 — 代数证明 + 对比表 + 代码

- **版式**：上中下三段
- **内容**：
  - 上 · 代数证明（Online Softmax 精确递推）：m_global=max_t m_t，s_global=Σ_t s_t·exp(m_t-m_global)，logΣexp(h)=m_global+log(s_global)。没有近似
  - 中 · 优化优势对比：

| 维度 | 传统 | CODA |
|------|------|------|
| HBM 写入量 | 完整 Logits（500MB-1GB） | 分块统计量（总计几 KB） |
| 显存占用 | 随词表增长极易 OOM | 近乎为零（不保存 Logits） |
| HBM 读取 | Softmax kernel 重新读完整 Logits | 辅助 kernel 只读几 KB |
| Kernel launch 数 | 2 | 2（但第二个 kernel I/O 量差 5 个数量级） |

  - 下 · 代码：`GemmLSESelectLogits = compose([LSE(), SelectLogits()]).bind(name="GemmLSESelectLogits", gemm_cls=GemmSm90)`
- **配图**：无（代码页）

---

## 第 20 页 · 过渡页 — §6 反向传播

- **版式**：章节过渡页
- **内容**：
  - §6
  - 反向传播
  - Theorem 1 打通了"前向 tile-local"到"反向自动融合"的逻辑链条
- **配图**：无

---

## 第 21 页 · §6.1 — 最简模型推导（前向→反向三步）

- **版式**：上公式下文
- **内容**：
  - 前向（最简模型）：h = xW₀，h' = f(h)，y = h'W₁
  - 反向（链式法则三步推导）：
    - Step 1：∇h'L = ∇yL·W₁ᵀ——标准 GEMM
    - Step 2：∇hL = ∇h'L ⊙ f'(h)——逐元素 Hadamard 积（tile-local）
    - Step 3：∇xL = ∇hL·W₀ᵀ——标准 GEMM
  - 结构对称表：

| | Step 1 | Step 2 | Step 3 |
|------|--------|--------|--------|
| 前向 | GEMM (W₀) | 本地变换 f(h) | GEMM (W₁) |
| 反向 | GEMM (W₁ᵀ) | 本地变换 ⊙f'(h) | GEMM (W₀ᵀ) |

  - 它们拥有完全一模一样的拓扑结构：GEMM → tile-local 变换 → GEMM。唯一区别是融合方向——前向 f 在 W₀ 的 Epilogue（前融），反向 ⊙f'(h) 在 W₀ᵀ 的 Epilogue（后融）
- **配图**：无

---

## 第 22 页 · §6.1 — 雅可比证明 + 图 9（图随文走）

- **版式**：左文右图
- **内容**：
  - 左：
    - Theorem 1 推广到 L 层：前向 h_ℓ=x_{ℓ-1}W_ℓ，x_ℓ[i,j]=f_ℓ[i,j](h_ℓ[i,j])。反向 ∇x_{ℓ-1}L=∇h_ℓL·W_ℓᵀ（GEMM），∇h_{ℓ-1}L[i,j]=g_{ℓ-1}[i,j](∇x_{ℓ-1}L[i,j], h_{ℓ-1}[i,j])（tile-local Epilogue）
    - 证明直觉：f_ℓ 仅依赖当前分块 [i,j] → 雅可比是分块对角矩阵 → 转置也是分块对角 → g_ℓ 天然 tile-local → 反向没有引入新的跨分块通信。计算 ∇h_{ℓ-1}L[i,j] 所需的所有材料，天然就在当前 GEMM 的寄存器里
  - 右：论文图 9（前向后向对称融合）
- **配图**：`09-fig9-fwd-bwd.png`

---

## 第 23 页 · §6.1 — 两个落地价值 + 💡 个人思考

- **版式**：左右双栏 + 底部标注
- **内容**：
  - 左 · 价值一——理论的普适性：SwiGLU（降维）、RoPE（保维）、各种激活函数反向梯度（升维）——每种都必须单独论证融合可行性。T1 用雅可比矩阵一言蔽之：只要前向 tile-local，反向必然 tile-local。不再有特例
  - 右 · 价值二——工程的自动化：T1 不意味着编译器能自动求导（silu'(x) 仍需手写），但它保证反向一定可写成 GEMM+Epilogue。开发者从"设计完整独立 CUDA kernel"降维为"写 tile-local VJP 函数"。FlashAttention 反向等了几个月，CODA 框架下写完 forward 当天就能跑
  - 对比表：

| | 传统手写融合 | CODA + T1 |
|------|------------|-----------|
| 反向 kernel 架构 | 从零设计 | 复用 forward 模板 |
| 融合可行性 | 每个函数单独论证 | T1 一劳永逸 |
| 实际要写什么 | 完整独立 CUDA kernel | tile-local VJP 函数 |

  - 💡 个人核心判断：T1 的真正戏眼，是用一个定理打通了"前向分块本地"到"反向自动融合"的逻辑链条。它把手艺人精雕细琢写算子的落后生产力，解放为编译器全自动生成的工业化通用范式
- **配图**：无

---

## 第 24 页 · §6.2 — RMSNorm 反向：两个 Problem + 恒等式

- **版式**：上文下表
- **内容**：
  - 连续计算流：h₀=xW₀ → h₁=f(h₀) → h₂=RMSNorm(h₁,γ) → y=h₂W₁。RMSNorm 反向引入两种打破 T1 的跨分块规约：
  - Problem A——行向统计量 s：s = (1/d)·sumcols(∇h₂L ⊙ h₂)。∇h₂L 和 h₂ 不在同一个 GEMM 寄存器边界上相遇
  - Problem B——γ 权重梯度的跨行规约：∇γL = Σ_rows ∇h₂L ⊙ h₁/rms(h₁)
  - 传统做法：启动独立 RMSNorm backward kernel → 从 HBM 重新读取 h₁ 和 ∇h₂L → 打破 CODA"零中间张量"原则
- **配图**：无

---

## 第 25 页 · §6.2 — CODA 解法 + 对比表

- **版式**：上中下三段
- **内容**：
  - 上 · Problem A 解法——恒等变换"位移"统计量：s = (1/d)·sumcols(∇h₂L ⊙ h₂) = (1/d)·sumcols(∇yL ⊙ y)（利用 ∇h₂L=∇yL·W₁ᵀ 和 y=h₂W₁）。硬件含义：y 和 ∇yL 恰好同时驻留在当前 GEMM 寄存器中 → Epilogue 顺手取 Hadamard 积 → 行向累加 → 只写回几个标量
  - 中 · Problem B 解法——分块局部规约 + 辅助合并：每个 GEMM 的 Epilogue 算本地分块局部累加 → 写出分块结果 → 辅助 kernel 跨行合并
  - 下 · 对比表：

| | 传统 | CODA |
|------|------|------|
| Problem A（行向 s） | 独立 kernel 读取 h₂+∇h₂L | 恒等式位移到寄存器边界 |
| Problem B（γ 梯度） | 同上 | 分块局部规约 + 辅助 kernel |
| HBM 额外读取 | 完整 h₂ 和 ∇h₂L 张量 | 仅分块统计量（几个标量） |

  - 💡 核心哲学：遇到全局规约，先尝试代数等价变换，变换不了再用两级规约——这是通用方法论，不是特例修补
- **配图**：无

---

## 第 26 页 · §6.3 — 代码展示：前后向对称 API

- **版式**：左边代码 + 右边讲解
- **内容**：
  - 左 · 代码：

```python
class LinearSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        pre_act, out = gemm_swiglu(x, weight.mT)  # GEMM + SwiGLU Epilogue
        ctx.save_for_backward(x, weight, pre_act)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight, pre_act = ctx.saved_tensors
        grad_pre = dswiglu_backward(pre_act, dout)  # VJP 也跑在 Epilogue 里
        dx = gemm(grad_pre, weight)                  # 标准 GEMM
        dweight = gemm(grad_pre.mT, x)               # 标准 GEMM
        return dx, dweight
```

  - 右 · 关键点：前向 gemm_swiglu 和反向 dswiglu_backward 都是 GEMM + Epilogue 结构。dswiglu_backward 虽然单独实现，但跑在 GEMM Epilogue 中而非独立 kernel launch。T1 保证了它一定可以融合——不需要为它设计新 kernel 架构。回顾 §3 的困境：反向必须重新手写完整独立 kernel → CODA 降维为手写 tile-local VJP
- **配图**：swiglu.py 代码截图

---

## 第 27 页 · 过渡页 — §7 实现

- **版式**：章节过渡页
- **内容**：
  - §7
  - 实现：从原语到硬件 & LLM 辅助编写
  - 三句话收束全篇技术落地 + AI 写算子为什么能逼近物理极限
- **配图**：无

---

## 第 28 页 · §7.1 — 从原语到硬件 + 架构调用链

- **版式**：上表下图
- **内容**：
  - 上 · 三句话收束：

| 动作 | 实现方式 | 硬件路径 |
|------|---------|---------|
| 数据搬运 | 向量→Shared Memory 广播，分块→TMA 异步（与计算 overlap） | HBM ↔ SM |
| 本地计算 | Pairwise Map 全在寄存器内，零线程通信 | Register |
| 规约 | Warp Shuffle → Shared Memory 合并 → 辅助 kernel | Register→SM→HBM |

  - 下 · 架构调用链：用户 API (swiglu.py) → _dispatch()（auto-tune tile size）→ cute.compile()（JIT: CuTe DSL→PTX→SASS）→ epilogue.visit()（寄存器内执行）→ compiled_fn(...)（GPU launch）
- **配图**：自绘架构调用链示意图

---

## 第 29 页 · §7.2 — LLM 为什么写不了 CUDA → CODA 如何降维

- **版式**：左右双栏
- **内容**：
  - 左 · 传统 LLM 写 CUDA 为什么翻车：
    - 搜索空间无限大：线程块/grid 大小、Shared Memory/寄存器分配——参数组合天文数字
    - 缺乏专家反馈：榨干 Hopper 的 GEMM 主循环需专家数周/数月调优（流水线、双缓冲、TMA 时序）。LLM 无法无中生有
  - 右 · CODA 的降维策略：
    - 固化 GEMM 主循环（专家设计，LLM 不能碰）
    - LLM 任务 = 从五类原语中选几个 + 指定拼接顺序
    - 精选示例库（Few-Shot，CuTeDSL 太新 LLM 没见过 → 照葫芦画瓢）
    - 声明式组合：`[Residual]→[RMSNorm γ]→[SwiGLU]` → 库自动嵌入 GEMM 尾部
  - 底部：这就像 CODA 已经盖好了钢筋混凝土大楼（GEMM 主循环），LLM 只需要决定房间怎么粉刷（Epilogue 原语组合）
- **配图**：无

---

## 第 30 页 · §7.2 — 图 10：LLM vs Human 性能对比（图随文走）

- **版式**：左图右文
- **内容**：
  - 左：论文图 10（LLM vs Human 性能对比——全文最有冲击力的实验结果之一）
  - 右 · 三个关键发现：
    - 橙条 ≈ 蓝条：LLM 生成的算子性能逼近甚至超过人类专家。反向场景中因更细粒度尝试反而略胜
    - 橙条 → 灰条：融合算子逼近纯矩阵乘法理论物理上限——Epilogue 开销几乎被完全隐藏
    - 工程范式验证："限定边界 + 原语组合"让 AI 从"大概率写出 bug 和慢代码"变成"稳定产出高性能算子"
  - 💡 论文作者用的是 Claude Code——不是因为它会写 CUDA，而是因为 CODA 把问题从"写 CUDA"变成了"选原语"
- **配图**：`10-fig10-llm-human.png`

---

## 第 31 页 · 过渡页 — §8 实验结果与成果展示

- **版式**：章节过渡页
- **内容**：
  - §8
  - 实验结果与成果展示
  - 论文已有结论 + 我的消融实验：将隐式设计决策显式量化
- **配图**：无

---

## 第 32 页 · §8.1 — 论文已有结论 + 图 11（图随文走）

- **版式**：左文右图
- **内容**：
  - 左 · 论文已证实的结论：

| 指标 | 结果 |
|------|------|
| 前向 Block 级加速 | ~10-15% vs torch.compile+cuBLAS |
| 反向 Block 级加速 | ~5-8% |
| d 越小加速比越大 | Memory-bound 越高收益越大 |
| CODA (LLM) vs (Human) | AI 逼近甚至超过人类专家 |

  - 右：论文图 11（Block 级加速比，分 d=2048/4096/8192 三列）
- **配图**：`11-fig11-block-speedup.png`

---

## 第 33 页 · §8.2 — 我的实验：动机 + 方法 + 设计矩阵

- **版式**：上图下表
- **内容**：
  - 上 · 动机：论文图 2 底行融合块被 Attention 和 Reduction 硬性隔开——融合边界 = tile-locality 被打破的物理边界。作者通过工程直觉找到了最优停损点，但没有用数据画出来。我的实验把"融合深度→加速比"曲线显式化
  - 方法——反证法：不做精确控制（需 CuTeDSL），而是每个深度同时跑两版：
    - 融合版：torch.compile 自动融合
    - 未融合版：PyTorch eager mode（一个 op=一个 kernel=一次 HBM 写回）
    - 差值 = 融合收益
  - 下 · 实验设计矩阵：

| 深度 | 操作链 | eager 版 kernel 数 |
|------|--------|-------------------|
| L0 | x @ w | 1 |
| L1 | x @ w + residual | 2 |
| L2 | L1 → RMSNorm γ 缩放 | ~5-6 |
| L3 | L2 + SwiGLU | ~8-10 |
| L4 | L3 + RoPE | ~10-12 |

  - 配置：M=N=K=4096，BF16，30 warmup+100 benchmark，测延迟（CUDA Event）+ kernel launch 数（profiler）
- **配图**：无（文字页）

---

## 第 34 页 · §8.2 — 实验三张图 + 核心叙事（图随文走）

- **版式**：三图并排 + 底部叙事
- **内容**：
  - 图 1（左）· 延迟对比柱状图：每个深度 fused vs unfused 并排，L2 跳升（RMSNorm reduction 被融合），L3→L4 增速放缓
  - 图 2（中）· Kernel launch 数对比：eager L2=9 kernels vs fused L2=4 kernels，减少 56%
  - 图 3（右）· 双 Y 轴黄金融合点：加速比 + kernel 数 vs 融合深度，标注"论文图 2 融合块边界"
  - 底部核心叙事：曲线拐点恰好落在论文图 2 融合块的边界上。不是作者不想继续融合（比如把 Attention 也融进去），而是 Softmax 的全局规约天然打破 tile-locality。到拐点后编译器已无法继续减少 kernel launch 数，继续叠加 Epilogue 只会让单 kernel 越来越重而不再减少 HBM 往返。论文作者通过工程直觉做出了正确的停损决策，我们用数据把它显式化了
- **配图**：`fig1_latency.png` + `fig2_kernel_counts.png` + `fig3_golden_point.png`

---

## 第 35 页 · 过渡页 — §9 局限性与思辨

- **版式**：章节过渡页
- **内容**：
  - §9
  - 局限性与思辨
  - 这套方法论能走多远？
- **配图**：无

---

## 第 36 页 · §9 — 四个局限 + CODA & FA 接力

- **版式**：四格卡片 + 底部示意
- **内容**：
  - ① 硬件锁定：强依赖 Hopper 架构 TMA 和累加器布局暴露。A100 无 TMA、消费级显卡累加器不开放、AMD 无 CUTLASS→无法直接移植
    - 💡 不止是"兼容性限制"——这暴露了 CODA 设计哲学的根本矛盾：体系结构定制 vs 通用性。累加器不向软件开放的 GPU 上，相邻元素可能分布在不同线程中→配对需要跨线程 shuffle→零开销优势被直接击穿
  - ② 算法局限：只适用于规整静态 GEMM。MoE 稀疏路由、动态 Token 丢弃打破 tile 假设
  - ③ 工程门槛：CuTeDSL 语法晦涩，即使有 LLM 辅助，普通工程师难以维护
  - ④ CODA vs FlashAttention：不是替代，是互补接力。CODA 负责层首尾（投影+归一化+激活），FA 攻坚中场（Attention+Softmax 全局规约）。两者联手=整层 Transformer 只需极少几次 HBM 读写
  - 底部：CODA + FlashAttention 接力示意图（自绘）
- **配图**：自绘 CODA+FA 接力示意图

---

## 第 37 页 · 过渡页 — §10 未来展望

- **版式**：章节过渡页
- **内容**：
  - §10
  - 未来展望
  - 从"手写算子"到"编译器自动合成"
- **配图**：无

---

## 第 38 页 · §10 — 四大方向 + 💡 DiT 思考

- **版式**：四段纵向排列
- **内容**：
  - ① 软件与芯片协同设计（Co-Design）：两大趋势
    - 从"手写算子"走向"编译器自动合成"：CODA 证明"固化 GEMM + 开放原语"可让 AI 自动生成硬件级算子。未来算法工程师只需写高层 Python，寄存器分配和 TMA 调度由 AI 编译器搞定
    - "以计算换带宽"思想的泛化：Memory Wall 越来越严重，"万物皆可 GEMM+Epilogue"会扩展到 SSM（Mamba）、扩散模型等
  - ② 学习路线（给想切入这个领域的同学）：掌握 Hopper/Blackwell TMA 和累加器布局 → 从 CUTLASS 3.x / CuTe 入手（Layout+Copy 机制是看懂 CODA 源码的钥匙）→ 探索 LLM 辅助编译（将硬件物理约束作为 Prompt 输入代码模型）
  - ③ 三个高落地价值的工业场景：万卡大规模训练（微小延迟累加=百万美元级浪费）/ 端侧设备（零拷贝闭环匹配端侧推理和 LoRA）/ 定制化 ASIC 生态（芯片厂商固化自己的 GEMM，算子层交给原语体系）
  - ④ 💡 我的思考——DiT 才是更值得的方向：
    > Transformer 已被 FlashAttention、Liger Kernel 等手写算子榨干到 90% 以上。CODA 在 Transformer 上只是"锦上添花"。而 DiT（扩散 Transformer）正处于"空有强大架构，极度缺乏系统级融合优化"的荒芜期——AdaLN 调制、时间步注入等胶水算子占 30-40% 耗时。设计 AdaLN-Epilogue 原语 + 多步去噪上下文片上常驻机制，将 CODA 推广至 DiT 多步推理，有望实现 1.5×-2× 的端到端加速
- **配图**：无

---

## 第 39 页 · 收尾

- **版式**：S09 Closing Manifesto（左 IKB 大字 + 右白底 takeaway）
- **内容**：
  - 左 · 大字：多用代数变换，少碰一次 HBM
  - 右 · 三条 Takeaway：
    - 01: GEMM+Epilogue 是通用范式——Transformer 中除 Attention 外的几乎所有操作可统一为该结构
    - 02: Theorem 1 打通了前后向自动融合——反向开发从"设计新 kernel"降维为"填空 VJP"
    - 03: LLM + 受限原语 = 工业化算子生产——从"手艺人精雕细琢"走向"编译器全自动生成"
- **配图**：无

---

## 汇总

| 项目 | 数量 |
|------|------|
| 总页数 | **39 页**（封面 1 + 过渡页 10 + 正文 27 + 收尾 1） |
| 章节过渡页 | 10 页（§1~§10，每节入口一页 IKB 满屏） |
| 论文原图 | 图 1-图 11（全部覆盖，图随文走不跨页） |
| 自绘图 | 五类原语速查表、架构调用链、CODA+FA 接力示意 |
| 原创实验图 | fig1 延迟对比 + fig2 kernel 数 + fig3 黄金融合点（3 张） |
| 💡 个人思考标记 | §1、§5.1（与 5.2/5.3 区别）、§6.1（T1 核心判断）、§6.2（通用方法论）、§7.2（Claude Code 选原语）、§9（体系结构定制 vs 通用性）、§10（DiT 蓝海）共 7 处 |
| 代码展示 | ops.py（§5.1）、registry.py + activation.py（§5.2）、registry.py（§5.3）、swiglu.py（§6.3） |
| 对比表 | §5.1/§5.2/§5.3/§6.1/§6.2/§8.1 共 6 张 |
