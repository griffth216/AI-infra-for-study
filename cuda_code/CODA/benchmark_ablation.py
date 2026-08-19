"""
消融实验：Epilogue 融合深度 vs 性能收益
==========================================
验证论文中算子融合块边界的设计决策 —— 用自动算子融合 vs eager mode 未融合，
量化不同融合深度下的加速比，画出"融合深度 → 加速比"曲线。

对应：notes/week4/消融实验设计.md

融合方案说明：
  本实验使用 torch.jit.script (TorchScript) 作为融合编译器。
  - TorchScript 内置的 TensorExpr fuser 会将 element-wise op 链融合为单个 CUDA kernel
    （例如：add + pow + sqrt + rsqrt + mul 融合为 fused_add_pow 等）
  - GEMM op 由 cuBLAS 单独执行（不在融合范围内）
  - 对比 eager mode（每个 op 独立 launch kernel），量化融合收益

  为什么不用 torch.compile？
    torch.compile 的 inductor 后端在 CUDA 上依赖 Triton，而 Triton 目前不支持 Windows
    原生运行。如果将来需要在 Linux/WSL2 上运行，可以将 @torch.jit.script 替换为
    @torch.compile 即可获得更强的融合效果。
"""

import json
import os
import torch


# ============================================================================
# 全局配置
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
M, N, K = 4096, 4096, 4096
WARMUP = 30
REPEATS = 100
EPS = 1e-6

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(OUTPUT_DIR, "ablation_results.json")

if DEVICE != "cuda":
    print("[WARN] CUDA 不可用，回退到 CPU。Benchmark 数据不具参考意义。")


# ============================================================================
# 2. Benchmark 工具
# ============================================================================
def benchmark_fn(fn, name, warmup=WARMUP, repeats=REPEATS):
    """warmup + CUDA event 计时，返回中位延迟 (ms)"""
    # warmup（首次调用会触发 JIT 编译）
    for _ in range(warmup):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # 正式计时
    timings = []
    if DEVICE == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(repeats):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
    else:
        import time
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            timings.append((time.perf_counter() - t0) * 1000.0)

    median_ms = sorted(timings)[len(timings) // 2]
    print(f"  {name:24s}: {median_ms:.4f} ms")
    return median_ms


# ============================================================================
# 3. 输入生成
# ============================================================================
def make_inputs(m=M, n=N, k=K):
    """生成各深度所需的共享输入张量"""
    x = torch.randn(m, k, device=DEVICE, dtype=DTYPE)
    w = torch.randn(k, n, device=DEVICE, dtype=DTYPE)
    residual = torch.randn(m, n, device=DEVICE, dtype=DTYPE)

    # RMSNorm weight（向量，维度 n）
    gamma = torch.randn(n, device=DEVICE, dtype=DTYPE)

    # SwiGLU 权重: gate/up 合并 (N × 2N)
    w_gate_up = torch.randn(n, 2 * n, device=DEVICE, dtype=DTYPE)

    # RoPE cos/sin 表 —— 与 paired-dimension 匹配：(M, N//2)
    cos = torch.randn(m, n // 2, device=DEVICE, dtype=DTYPE)
    sin = torch.randn(m, n // 2, device=DEVICE, dtype=DTYPE)

    return x, w, residual, gamma, w_gate_up, cos, sin


# ============================================================================
# 4. Eager 版（未融合基线）—— 每个 op 独立 launch kernel
# ============================================================================

# L0: 纯 GEMM
def eager_l0(x, w):
    return x @ w


# L1: GEMM + Residual
def eager_l1(x, w, residual):
    h = x @ w            # kernel 1: cuBLAS GEMM
    h = h + residual     # kernel 2: element-wise add
    return h


# L2: GEMM + Residual + RMSNorm
def eager_l2(x, w, residual, gamma, eps=EPS):
    h = x @ w                                  # kernel 1: GEMM
    h = h + residual                           # kernel 2: add
    h_sq = h ** 2                              # kernel 3: square
    var = torch.mean(h_sq, dim=-1, keepdim=True)  # kernel 4: reduction
    rstd = 1.0 / torch.sqrt(var + eps)         # kernel 5: rsqrt
    h = h * rstd                               # kernel 6: scale
    h = h * gamma                              # kernel 7: γ broadcast
    return h


# L3: L2 + SwiGLU
def eager_l3(x, w, residual, gamma, w_gate_up, eps=EPS):
    h = x @ w
    h = h + residual
    h_sq = h ** 2
    var = torch.mean(h_sq, dim=-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    h_norm = h * rstd * gamma

    # SwiGLU = silu(gate) * up
    gate_up = h_norm @ w_gate_up               # kernel: GEMM (M×N × N×2N)
    gate, up = gate_up.chunk(2, dim=-1)         # kernel: split/view
    activated = gate * torch.sigmoid(gate)       # kernel: silu
    out = activated * up                         # kernel: element-wise multiply
    return out


# L4: L3 + RoPE
def eager_l4(x, w, residual, gamma, w_gate_up, cos, sin, eps=EPS):
    h = x @ w
    h = h + residual
    h_sq = h ** 2
    var = torch.mean(h_sq, dim=-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    h_norm = h * rstd * gamma

    gate_up = h_norm @ w_gate_up
    gate, up = gate_up.chunk(2, dim=-1)
    activated = gate * torch.sigmoid(gate)
    out = activated * up

    # RoPE（旋转位置编码）
    out_2d = out.view(out.shape[0], -1, 2)
    out_rot = torch.empty_like(out_2d)
    out_rot[..., 0] = out_2d[..., 0] * cos - out_2d[..., 1] * sin
    out_rot[..., 1] = out_2d[..., 0] * sin + out_2d[..., 1] * cos
    return out_rot.view_as(out)


# ============================================================================
# 5. 融合版 —— 使用 torch.jit.script（TorchScript）
#
# TorchScript 的 TensorExpr fuser 会自动将多个 element-wise op 融合为单个 kernel。
# 在 Windows 上这是 torch.compile 的最佳替代方案。
#
# 验证方法：对融合版跑一次 profiler，在输出中查找 "fused_" 前缀的 kernel 名。
# ============================================================================

@torch.jit.script
def fused_l0(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x @ w


@torch.jit.script
def fused_l1(x: torch.Tensor, w: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    h = x @ w
    h = h + residual
    return h


@torch.jit.script
def fused_l2(x: torch.Tensor, w: torch.Tensor,
             residual: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    eps: float = 1e-6
    h = x @ w
    h = h + residual
    h_sq = h ** 2
    var = torch.mean(h_sq, dim=-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    h = h * rstd
    h = h * gamma
    return h


@torch.jit.script
def fused_l3(x: torch.Tensor, w: torch.Tensor, residual: torch.Tensor,
             gamma: torch.Tensor, w_gate_up: torch.Tensor) -> torch.Tensor:
    eps: float = 1e-6
    h = x @ w
    h = h + residual
    h_sq = h ** 2
    var = torch.mean(h_sq, dim=-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    h_norm = h * rstd * gamma
    gate_up = h_norm @ w_gate_up
    gate, up = gate_up.chunk(2, dim=-1)
    out_val = gate * torch.sigmoid(gate) * up
    return out_val


@torch.jit.script
def fused_l4(x: torch.Tensor, w: torch.Tensor, residual: torch.Tensor,
             gamma: torch.Tensor, w_gate_up: torch.Tensor,
             cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    eps: float = 1e-6
    h = x @ w
    h = h + residual
    h_sq = h ** 2
    var = torch.mean(h_sq, dim=-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    h_norm = h * rstd * gamma
    gate_up = h_norm @ w_gate_up
    gate, up = gate_up.chunk(2, dim=-1)
    out_val = gate * torch.sigmoid(gate) * up
    out_2d = out_val.view(out_val.shape[0], -1, 2)
    out_rot = torch.empty_like(out_2d)
    out_rot_0 = out_2d[..., 0] * cos - out_2d[..., 1] * sin
    out_rot_1 = out_2d[..., 0] * sin + out_2d[..., 1] * cos
    out_rot = torch.stack([out_rot_0, out_rot_1], dim=-1)
    return out_rot.view_as(out_val)


# ============================================================================
# 6. 主实验循环
# ============================================================================
def run_experiment():
    print("=" * 64)
    print("消融实验：Epilogue 融合深度 vs 性能收益")
    print(f"设备: {DEVICE}  |  dtype: {DTYPE}  |  (M,N,K) = ({M},{N},{K})")
    print(f"Warmup: {WARMUP}  |  Repeats: {REPEATS}")
    print(f"融合方式: torch.jit.script (TorchScript TensorExpr Fuser)")
    print("=" * 64)

    x, w, residual, gamma, w_gate_up, cos, sin = make_inputs()

    depths = ["L0", "L1", "L2", "L3", "L4"]
    results = {d: {"eager_ms": 0.0, "fused_ms": 0.0} for d in depths}

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    eager_funcs = {
        "L0": lambda: eager_l0(x, w),
        "L1": lambda: eager_l1(x, w, residual),
        "L2": lambda: eager_l2(x, w, residual, gamma),
        "L3": lambda: eager_l3(x, w, residual, gamma, w_gate_up),
        "L4": lambda: eager_l4(x, w, residual, gamma, w_gate_up, cos, sin),
    }

    fused_funcs = {
        "L0": lambda: fused_l0(x, w),
        "L1": lambda: fused_l1(x, w, residual),
        "L2": lambda: fused_l2(x, w, residual, gamma),
        "L3": lambda: fused_l3(x, w, residual, gamma, w_gate_up),
        "L4": lambda: fused_l4(x, w, residual, gamma, w_gate_up, cos, sin),
    }

    for depth in depths:
        print(f"\n--- {depth} ---")
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Eager（未融合基线）
        results[depth]["eager_ms"] = benchmark_fn(eager_funcs[depth], f"{depth} eager")

        # Fused（TorchScript 融合）
        results[depth]["fused_ms"] = benchmark_fn(fused_funcs[depth], f"{depth} fused")

    # =========================================================================
    # 汇总
    # =========================================================================
    print("\n" + "=" * 72)
    print(f"{'Depth':<8} {'Eager (ms)':<15} {'Fused (ms)':<15} {'Speedup':<10} {'Reduction %':<12}")
    print("-" * 72)
    for d in depths:
        e = results[d]["eager_ms"]
        f = results[d]["fused_ms"]
        speedup = e / f if f > 0 else 0.0
        reduction = (e - f) / e * 100 if e > 0 else 0.0
        results[d]["speedup"] = round(speedup, 3)
        results[d]["reduction_pct"] = round(reduction, 2)
        print(f"{d:<8} {e:<15.4f} {f:<15.4f} {speedup:<10.2f}x {reduction:<11.1f}%")
    print("=" * 72)

    # 保存结果到 JSON 供绘图脚本使用
    results["config"] = {
        "device": DEVICE,
        "dtype": str(DTYPE),
        "M": M, "N": N, "K": K,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "fusion_backend": "torch.jit.script (TorchScript TensorExpr Fuser)",
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    run_experiment()
