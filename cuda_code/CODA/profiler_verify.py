"""
Profiler 验证：Eager vs TorchScript Fused Kernel Launch 数量对比
=================================================================
对每个深度分别跑 eager 和 TorchScript 融合版本，用 torch.profiler 统计
CUDA kernel 数量，验证 TorchScript 的 TensorExpr fuser 确实减少了 kernel launch。

在 profiler 输出中查找：
  - Eager 版：多个独立的 element-wise kernel（add, pow, mul, sqrt 等）
  - Fused 版："fused_" 前缀的 kernel（如 fused_add_pow, fused_add_sqrt_reciprocal_mul_mul_mul）

输出：
  1. 终端表格（kernel 名称 + 耗时）
  2. Chrome trace JSON（用 chrome://tracing 打开可视化）
"""

import os
import torch
from benchmark_ablation import (
    DEVICE, DTYPE, M, N, K, EPS, make_inputs,
    eager_l0, eager_l1, eager_l2, eager_l3, eager_l4,
    fused_l0, fused_l1, fused_l2, fused_l3, fused_l4,
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACE_DIR = os.path.join(OUTPUT_DIR, "traces")
os.makedirs(TRACE_DIR, exist_ok=True)

DEPTHS = ["L0", "L1", "L2", "L3", "L4"]


def count_cuda_kernels(prof):
    """从 profiler key_averages 中统计有 CUDA 执行时间的 kernel 数量。

    使用 self_device_time_total > 0 过滤，确保只统计实际在 GPU 上执行的操作。
    """
    table = prof.key_averages()
    cuda_kernels = [row for row in table if getattr(row, "self_device_time_total", 0) > 0]
    return len(cuda_kernels)


def profile_depth(name, eager_fn, fused_fn, export_trace=True):
    print(f"\n{'='*60}")
    print(f"  Profiling: {name}")
    print(f"{'='*60}")

    # --- Eager ---
    print(f"\n  [{name}] Eager 版本 (未融合):")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof_eager:
        eager_fn()
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    n_eager = count_cuda_kernels(prof_eager)
    print(f"    → CUDA kernel 数量: {n_eager}")
    print(prof_eager.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    if export_trace:
        trace_path = os.path.join(TRACE_DIR, f"trace_{name}_eager.json")
        prof_eager.export_chrome_trace(trace_path)
        print(f"    Chrome trace 已保存: {trace_path}")

    # --- Fused ---
    print(f"\n  [{name}] Fused 版本 (TorchScript):")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof_fused:
        fused_fn()
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    n_fused = count_cuda_kernels(prof_fused)
    print(f"    → CUDA kernel 数量: {n_fused}")
    print(prof_fused.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    # 提示：在输出中搜索 "fused_" 开头的 kernel 名
    fused_kernel_names = [r.key for r in prof_fused.key_averages()
                          if getattr(r, "self_device_time_total", 0) > 0 and "fused" in r.key.lower()]
    if fused_kernel_names:
        print(f"    [FUSED] Detected fused kernels: {fused_kernel_names}")

    if export_trace:
        trace_path = os.path.join(TRACE_DIR, f"trace_{name}_fused.json")
        prof_fused.export_chrome_trace(trace_path)
        print(f"    Chrome trace 已保存: {trace_path}")

    return {"depth": name, "eager_kernels": n_eager, "fused_kernels": n_fused}


def run_profiler():
    print("=" * 64)
    print("Profiler 验证：Kernel Launch 数量对比")
    print(f"设备: {DEVICE}  |  dtype: {DTYPE}  |  (M,N,K) = ({M},{N},{K})")
    print(f"融合方式: torch.jit.script (TorchScript)")
    print("=" * 64)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    x, w, residual, gamma, w_gate_up, cos, sin = make_inputs()

    # 先 warmup 触发 JIT 编译（避免 profiler 包含编译开销）
    print("\n[Warmup] 触发 TorchScript 编译...")
    _ = fused_l4(x, w, residual, gamma, w_gate_up, cos, sin)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("[Warmup] 完成。\n")

    eager_fns = {
        "L0": lambda: eager_l0(x, w),
        "L1": lambda: eager_l1(x, w, residual),
        "L2": lambda: eager_l2(x, w, residual, gamma),
        "L3": lambda: eager_l3(x, w, residual, gamma, w_gate_up),
        "L4": lambda: eager_l4(x, w, residual, gamma, w_gate_up, cos, sin),
    }

    fused_fns = {
        "L0": lambda: fused_l0(x, w),
        "L1": lambda: fused_l1(x, w, residual),
        "L2": lambda: fused_l2(x, w, residual, gamma),
        "L3": lambda: fused_l3(x, w, residual, gamma, w_gate_up),
        "L4": lambda: fused_l4(x, w, residual, gamma, w_gate_up, cos, sin),
    }

    all_results = []
    for depth in DEPTHS:
        result = profile_depth(depth, eager_fns[depth], fused_fns[depth])
        all_results.append(result)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # =========================================================================
    # 汇总表
    # =========================================================================
    print("\n" + "=" * 72)
    print(f"{'Depth':<8} {'Eager Kernels':<16} {'Fused Kernels':<16} {'Reduction':<12}")
    print("-" * 72)
    for r in all_results:
        e = r["eager_kernels"]
        f = r["fused_kernels"]
        red = (e - f) / e * 100 if e > 0 else 0
        print(f"{r['depth']:<8} {e:<16} {f:<16} {red:<11.1f}%")
    print("=" * 72)
    print(f"\nChrome trace 文件保存在: {TRACE_DIR}")
    print("在 chrome://tracing 中打开 .json 文件即可查看 kernel timeline。")

    return all_results


if __name__ == "__main__":
    run_profiler()
