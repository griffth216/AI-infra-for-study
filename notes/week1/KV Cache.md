## 1. 从注意力机制说起：KV Cache 的由来
要理解 KV Cache，必须先回到 Transformer 的[自注意力机制](https://zhida.zhihu.com/search?content_id=271534792&content_type=Article&match_order=1&q=%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6&zhida_source=entity)。在自回归生成（autoregressive generation）过程中，模型每次只生成一个 token，然后将这个 token 拼接到已有序列后面，再生成下一个 token。这个过程看似简单，但隐藏着巨大的计算冗余。

在训练阶段，由于有 causal mask（因果掩码），整个序列的 Q、K、V 可以并行计算。但在推理的自回归生成阶段，当我们生成第 t个 token 时：
![[Pasted image 20260602125308.png]]
![[Pasted image 20260602125337.png]]
![[Pasted image 20260602130029.png]]

## 2. KV Cache 到底缓存了什么？数学推导

### 2.1 单层、单头的 KV Cache
![[Pasted image 20260602130158.png]]
### 2.2 整个模型的 KV Cache

![[Pasted image 20260602130240.png]]
### 2.3 为什么不缓存 Query？
![[Pasted image 20260602130513.png]]

## 3. 内存占用：为什么 KV Cache 会成为瓶颈
### 3.1 量化计算
为了极大地提升生成速度，大模型被迫采取了“以空间换时间”的策略，但这个“空间”的增长速度是极其惊人的。
KV Cache 的容量公式长这样：

$$\text{Cache Size} = 2 \times B \times L \times N \times H \times D \times \text{Bytes}$$
- **$2$**：因为要存 $K$ 和 $V$ 两个矩阵。
    
- **$B$ (Batch Size，并发数)**：如果你是 OpenAI，你不可能一次只服务一个用户。如果你的服务器同时在给 64 个人生成回答，每个人的缓存都要独立存放。
    
- **$L$ (Sequence Length，上下文长度)**：现在动辄支持 128K（十万字）的长文本阅读，这个数字非常大。
    
- **$N$ (Layers，层数)**：模型有几十层（比如 32 层），**每一层**都要存一份独立的 KV Cache！
    
- **$H$ (Heads，多头数) $\times$ $D$ (头维度)**：比如 32个头，每个头 128 维。
    
- **$\text{Bytes}$ (精度)**：通常使用半精度浮点数（FP16），每个数字占 2 个字节。

这是工程界最痛的点。KV Cache 带来的瓶颈分为两个层面：

1. **容量瓶颈（存不下）：** 显卡非常贵，显存容量是硬约束。KV Cache 太大，导致一张卡能同时服务的用户数量（Batch Size）极低，公司的算力成本无法摊薄。
    
2. **带宽瓶颈（搬不动 / 内存墙）：** 这是更致命的。在生成每一个 Token 时，GPU 的计算单元（Tensor Cores）其实运算极快，但它**必须把几十 GB 的 KV Cache 从显存（HBM）全部搬运到计算芯片里**。 目前 GPU 的显存读取速度（带宽）远远跟不上计算速度。结果就是：**计算单元大部分时间都在“发呆”等待数据传输。** 这种状态在工程上被称为 **Memory-bound（访存受限）**。

### 3.2 KV Cache 的动态性问题
与模型权重不同，KV Cache 有几个让它格外难处理的特点：

**动态增长**：每生成一个 token，KV Cache 就增长一块。这意味着很难在请求开始前就为它分配好内存。

**请求间差异巨大**：不同请求可能生成 10 个 token，也可能生成 10000 个 token。如果按最大长度预分配，内存浪费严重；如果动态分配，又容易产生碎片。

**批处理复杂性**：在一个 batch 中，不同请求处于不同的生成阶段，KV Cache 长度各不相同，难以高效地打包成规则张量。

这些问题在 vLLM 论文中有详细分析，实验数据显示，在朴素实现下，KV Cache 内存碎片导致的浪费可以高达 **60-80%** 的可用显存。

### 3.3 推理阶段的两个瓶颈

LLM 推理通常分为两个阶段：

- **Prefill（预填充）阶段**：并行处理输入 prompt，一次性计算所有输入 token 的 KV，填充 KV Cache。这个阶段是**计算密集型（compute-bound）**。
- **Decode（解码）阶段**：逐 token 生成，每步只处理一个 token，但需要读取全部 KV Cache。这个阶段是**内存带宽密集型（memory-bound）**。

在 Decode 阶段，GPU 的算力（FLOPS）大部分是空闲的，瓶颈在于把 KV Cache 从显存搬到计算单元的带宽。**这就是为什么优化 KV Cache 的内存布局和访问模式，比单纯提升算力更重要**。



## [(3 封私信 / 49 条消息) KV Cache 深度解析：从原理到工程优化的完整指南 - 知乎](https://zhuanlan.zhihu.com/p/2016843212178882587)
