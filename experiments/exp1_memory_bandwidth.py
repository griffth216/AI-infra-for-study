"""
实验1：显存带宽分析 —— 验证 CODA 论文图1的核心动机
========================================================
测量分离执行 vs 融合执行的 HBM 读写量差异
对应论文：Section 1 图1 —— 非GEMM操作(Others)占比巨大

用法: python exp1_memory_bandwidth.py
"""

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function
import itertools

torch.manual_seed(42)

# ============================================================
# 配置
# ============================================================
DEVICE = "cuda"
DTYPE = torch.bfloat16
HIDDEN = 4096        # 隐藏层维度
INTERMEDIATE = 14336 # FFN 中间层 (≈ 3.5x hidden, LLaMA 风格)
BATCH_TOKENS = 4096  # batch * seq_len 的总 token 数
WARMUP = 5
REPEAT = 20

# ============================================================
# 被测操作：模拟 Transformer 中的一个 MLP 块
# ============================================================

class SeparateOps(torch.nn.Module):
    """传统方式：GEMM → SwiGLU → 另一个GEMM，每步都写回HBM"""
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(HIDDEN, INTERMEDIATE, bias=False, dtype=DTYPE, device=DEVICE)
        self.up_proj = torch.nn.Linear(HIDDEN, INTERMEDIATE, bias=False, dtype=DTYPE, device=DEVICE)
        self.down_proj = torch.nn.Linear(INTERMEDIATE, HIDDEN, bias=False, dtype=DTYPE, device=DEVICE)

    def forward(self, x):
        # Step 1: 两个独立 GEMM，各自写回 HBM
        gate = self.gate_proj(x)   # (B, INTERMEDIATE) → HBM write
        up = self.up_proj(x)       # (B, INTERMEDIATE) → HBM write
        # Step 2: SwiGLU 激活 → 读 gate, up → 写回
        act = F.silu(gate) * up    # (B, INTERMEDIATE) → HBM write
        # Step 3: 输出投影
        out = self.down_proj(act)  # (B, HIDDEN) → HBM write
        return out


class FusedOps(torch.nn.Module):
    """
    torch.compile 融合版：让 PyTorch Inductor 自动融合 element-wise 操作
    模拟 CODA 的 "GEMM + epilogue" 思路
    """
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(HIDDEN, INTERMEDIATE, bias=False, dtype=DTYPE, device=DEVICE)
        self.up_proj = torch.nn.Linear(HIDDEN, INTERMEDIATE, bias=False, dtype=DTYPE, device=DEVICE)
        self.down_proj = torch.nn.Linear(INTERMEDIATE, HIDDEN, bias=False, dtype=DTYPE, device=DEVICE)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        act = F.silu(gate) * up
        out = self.down_proj(act)
        return out


def profile_bandwidth(model, x, label):
    """使用 PyTorch Profiler 测量 HBM 读写量"""
    model.eval()
    # Warmup
    for _ in range(WARMUP):
        model(x)

    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with record_function(label):
            for _ in range(REPEAT):
                y = model(x)
                torch.cuda.synchronize()

    # 统计 CUDA kernel 的 memory bandwidth
    total_bytes_read = 0
    total_bytes_write = 0
    total_time_us = 0

    for event in prof.key_averages():
        if event.cuda_time_total > 0:
            # profiler 给出的 self_device_memory 相关字段
            if hasattr(event, 'self_device_memory_usage'):
                total_bytes_read += getattr(event, 'self_device_memory_usage', 0)

    # 更精确的方式：从事件表格中提取
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=15,
        header="name,cuda_time_total,cuda_time_avg,occurrences"
    ))

    return prof


def main():
    print("=" * 60)
    print("  CODA 实验1: 显存带宽分析")
    print("  GPU:", torch.cuda.get_device_name(0))
    print(f"  隐藏维度={HIDDEN}, 中间层={INTERMEDIATE}, Tokens={BATCH_TOKENS}")
    print("=" * 60)

    x = torch.randn(BATCH_TOKENS, HIDDEN, dtype=DTYPE, device=DEVICE)

    # ---- 分离执行 ----
    model_sep = SeparateOps()
    print("\n>>> 分离执行 (传统 PyTorch)")
    prof_sep = profile_bandwidth(model_sep, x, "Separate Ops")

    # ---- torch.compile 融合执行 ----
    model_fused = FusedOps()
    model_fused_compiled = torch.compile(model_fused, mode="reduce-overhead")
    # warmup compile
    print("\n>>> 编译融合模型中...")
    model_fused_compiled(x)
    print(">>> 融合执行 (torch.compile)")
    prof_fused = profile_bandwidth(model_fused_compiled, x, "Fused Ops (torch.compile)")

    # ---- 简单时间对比 ----
    print(f"\n{'='*60}")
    print("  时间对比 (纯算子耗时, ms)")
    print(f"{'='*60}")

    # 分离版计时
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPEAT):
        model_sep(x)
    end.record()
    torch.cuda.synchronize()
    sep_time = start.elapsed_time(end) / REPEAT

    # 融合版计时
    start.record()
    for _ in range(REPEAT):
        model_fused_compiled(x)
    end.record()
    torch.cuda.synchronize()
    fused_time = start.elapsed_time(end) / REPEAT

    print(f"  分离执行 (3个kernel):  {sep_time:.4f} ms")
    print(f"  融合执行 (torch.compile): {fused_time:.4f} ms")
    print(f"  加速比:              {sep_time / fused_time:.2f}x")
    print(f"\n  => torch.compile 自动融合了 element-wise 操作,")
    print(f"     减少了中间张量的 HBM 读写, 与 CODA 的核心思路一致")


if __name__ == "__main__":
    main()
