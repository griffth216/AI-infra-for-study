
import torch
import math


def standard_attention(Q, K, V):
    """
    标准 Scaled Dot-Product Attention
    Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d)) @ V

    问题：Q@K^T 产生 [N,N] 矩阵 S，softmax 又产生 [N,N] 矩阵 P
    N 大时这两个矩阵直接撑爆显存
    """
    d = Q.shape[-1]
    S = (Q @ K.T) / math.sqrt(d)   # [N, N] ← 罪魁祸首
    P = torch.softmax(S, dim=-1)   # [N, N] ← 又一个
    O = P @ V
    return O, S, P


# ═══════════════════════════════════════════════════════════════════════════════
# FlashAttention 前向传播（笔记第三节伪代码的逐行翻译）
# ═══════════════════════════════════════════════════════════════════════════════

def flash_attention(Q, K, V, block_size=64):
    """
    FlashAttention-2 风格的实现。

    外层遍历 Q 块 → Q 块锁定在"SRAM"中
    内层遍历 K,V 块 → 逐个读入，算完就扔

    三个"储物格"（SRAM 中维护的状态变量）：
      m  — 全局最大值（用于安全 softmax 的缩放基准）
      l  — 分母累加和
      Õ  — 未归一化输出累加（分子部分）

    输入: Q, K, V ∈ [N, d]
    输出: O ∈ [N, d]
    """
    N, d = Q.shape
    O = torch.zeros(N, d)  # 最终输出（存在"HBM"中）

    # ── 外层循环：遍历 Q 块 ──
    for i_start in range(0, N, block_size):
        i_end = min(i_start + block_size, N)
        Qi = Q[i_start:i_end]                  # 从 HBM 读入一个 Q 块
        Br = Qi.shape[0]                       # 这个块有多少行

        # ★ 三个储物格，初始值 ★
        m = torch.full((Br,), -float('inf'))   # 储物格 1: 全局 max
        l = torch.zeros((Br,))                 # 储物格 2: 分母
        O_block = torch.zeros((Br, d))         # 储物格 3: 未归一化输出

        # ── 内层循环：遍历 K,V 块 ──
        for j_start in range(0, N, block_size):
            j_end = min(j_start + block_size, N)
            Kj = K[j_start:j_end]              # 从 HBM 读入 K 块
            Vj = V[j_start:j_end]              # 从 HBM 读入 V 块

            # 步骤 1: 局部注意力分数 S_ij = Qi @ Kj^T / sqrt(d)
            S = (Qi @ Kj.T) / math.sqrt(d)     # [Br, Bc] — 在 SRAM 中

            # 步骤 2: Online Softmax 更新
            m_new = torch.max(m, S.max(dim=-1).values)       # 新的全局 max
            scale = torch.exp(m - m_new)                      # 缩放因子：旧→新基准
            l = l * scale + torch.exp(S - m_new.unsqueeze(-1)).sum(dim=-1)

            # 步骤 3: 输出更新（旧的也要按同样比例缩放）
            O_block = O_block * scale.unsqueeze(-1) + \
                      torch.exp(S - m_new.unsqueeze(-1)) @ Vj

            m = m_new
            # ★ S, Kj, Vj 在这里被丢弃，不写回显存！★

        # 所有 K,V 块遍历完 → 最终归一化 → 唯一一次写入 HBM
        O[i_start:i_end] = O_block / l.unsqueeze(-1)

    return O


# ═══════════════════════════════════════════════════════════════════════════════
# 验证：正确性 + 显存对比
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)

    N, d = 512, 64
    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)

    print("=" * 55)
    print("FlashAttention 核心实现验证")
    print("=" * 55)
    print(f"  序列长度 N = {N}, 头维度 d = {d}")

    # 1. 数学正确性
    O_std, S, P = standard_attention(Q, K, V)
    O_flash = flash_attention(Q, K, V, block_size=64)

    err = (O_std - O_flash).abs().max().item()
    print(f"\n  [正确性] 标准 vs Flash 最大误差: {err:.2e}")

    # 2. 中间矩阵大小对比
    middle_matrices_mb = 2 * N * N * 4 / (1024 * 1024)   # S + P, float32
    states_mb = 3 * N * d * 4 / (1024 * 1024)             # m + l + O, float32
    print(f"\n  [显存] 标准 Attention 中间矩阵: {middle_matrices_mb:.1f} MB (S+P, 各 {N}×{N})")
    print(f"         FlashAttention 状态变量: {states_mb:.2f} MB (m + l + O)")
    print(f"         节省: {middle_matrices_mb / states_mb:.0f}×")

    # 3. 如果 N=4096 呢？
    N_large = 4096
    large_mid = 2 * N_large * N_large * 4 / (1024 * 1024)
    large_fa = 3 * N_large * d * 4 / (1024 * 1024)
    print(f"\n  [如果 N=4096] 标准中间矩阵: {large_mid:.0f} MB")
    print(f"              FlashAttention: {large_fa:.1f} MB")
    print(f"              节省: {large_mid / large_fa:.0f}×")
    print(f"              → 标准方案单头就占 {large_mid:.0f} MB，32 头 = {large_mid*32:.0f} MB = {large_mid*32/1024:.1f} GB")

    print(f"\n  [核心公式] 笔记第二节的 Online Softmax 更新:")
    print(f"    m_new = max(m_old, x_new)")
    print(f"    l_new = l_old × e^(m_old - m_new)  +  e^(x_new - m_new)")
    print(f"    O_new = O_old × e^(m_old - m_new)  +  e^(x_new - m_new) × V")
    print(f"            ↑缩放旧结果到新基准          ↑加入新块的贡献")
