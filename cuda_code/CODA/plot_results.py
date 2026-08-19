"""
绘制消融实验结果图
==================
读入 benchmark_ablation.py 和 profiler_verify.py 的输出，
生成 3 张图：
  图1: 延迟对比柱状图（Fused vs Unfused）
  图2: Kernel Launch 数量对比
  图3: 双 Y 轴合成图 — "黄金融合点"

用法：
  python plot_results.py                    # 只生成图表
  python plot_results.py --results <json>   # 指定结果文件
"""

import json
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# 尝试使用中文字体
try:
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(OUTPUT_DIR, "ablation_results.json")

# 模拟的 profiler 数据（用户也可以自行填写实际 profiler 结果）
# 实测数据 (RTX 4060, BF16, TorchScript) —— profiler 统计唯一 CUDA kernel 类型
DEFAULT_PROFILER = {
    "L0": {"eager_kernels": 1, "fused_kernels": 1},    # 纯 GEMM，无融合空间
    "L1": {"eager_kernels": 2, "fused_kernels": 2},    # GEMM + add（add 太小无法单独融合）
    "L2": {"eager_kernels": 9, "fused_kernels": 4},    # fused_add_pow + fused_add_sqrt_reciprocal_mul_mul_mul
    "L3": {"eager_kernels": 12, "fused_kernels": 8},   # SwiGLU 融合
    "L4": {"eager_kernels": 13, "fused_kernels": 9},   # RoPE 融合，但减速
}


def load_results(path):
    """加载 benchmark 结果 JSON"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def plot_latency_bars(results, save_path):
    """图 1: 延迟对比柱状图（Fused vs Unfused）"""
    depths = ["L0", "L1", "L2", "L3", "L4"]
    if "config" in results:
        del results["config"]

    eager_ms = [results[d]["eager_ms"] for d in depths]
    fused_ms = [results[d]["fused_ms"] for d in depths]
    speedups = [results[d].get("speedup", 0) for d in depths]

    x = np.arange(len(depths))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, eager_ms, width, label="Eager (unfused)", color="#D84B4B", edgecolor="white")
    bars2 = ax.bar(x + width / 2, fused_ms, width, label="Fused (torch.compile)", color="#4B7FD8", edgecolor="white")

    # 柱顶标注加速比
    for i, (e, f, s) in enumerate(zip(eager_ms, fused_ms, speedups)):
        ax.text(i, max(e, f) + max(eager_ms) * 0.01, f"{s:.1f}×", ha="center", fontsize=10, fontweight="bold", color="#333")

    ax.set_xlabel("Fusion Depth", fontsize=13)
    ax.set_ylabel("Latency (ms)", fontsize=13)
    ax.set_title("图 1: Latency Comparison — Eager vs Fused (torch.compile)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(depths, fontsize=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(eager_ms) * 1.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"图 1 已保存: {save_path}")
    plt.close(fig)


def plot_kernel_counts(profiler_data, save_path):
    """图 2: Kernel Launch 数量对比"""
    depths = ["L0", "L1", "L2", "L3", "L4"]
    eager_k = [profiler_data[d]["eager_kernels"] for d in depths]
    fused_k = [profiler_data[d]["fused_kernels"] for d in depths]

    x = np.arange(len(depths))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, eager_k, width, label="Eager Kernels", color="#D84B4B", edgecolor="white")
    bars2 = ax.bar(x + width / 2, fused_k, width, label="Fused Kernels (torch.compile)", color="#4B7FD8", edgecolor="white")

    # 柱顶标数字
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(int(bar.get_height())), ha="center", fontsize=10, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(int(bar.get_height())), ha="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Fusion Depth", fontsize=13)
    ax.set_ylabel("CUDA Kernel Count", fontsize=13)
    ax.set_title("图 2: Kernel Launch Count — Eager vs Fused", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(depths, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(eager_k) * 1.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"图 2 已保存: {save_path}")
    plt.close(fig)


def plot_golden_fusion_point(results, profiler_data, save_path):
    """图 3: 双 Y 轴合成图 —— "黄金融合点"

    Y1 (柱状图): 加速比 (eager / fused)
    Y2 (折线):   fused 版 kernel launch 数
    X 轴:       融合深度 L0 → L4

    在曲线拐点处标注论文图 2 的融合边界。
    """
    depths = ["L0", "L1", "L2", "L3", "L4"]
    if "config" in results:
        del results["config"]

    speedups = [results[d].get("speedup", 0) for d in depths]
    fused_kernels = [profiler_data[d]["fused_kernels"] for d in depths]

    x = np.arange(len(depths))
    colors_bar = ["#E8A87C", "#F4C7AB", "#D4A574", "#C4945E", "#B8844A"]

    fig, ax1 = plt.subplots(figsize=(11, 7))

    # Y1: 加速比柱状图
    bars = ax1.bar(x, speedups, width=0.5, color=colors_bar, edgecolor="white", zorder=3)
    ax1.set_ylabel("Speedup (Eager / Fused)", fontsize=13, color="#C45A3C")
    ax1.set_ylim(0, max(speedups) * 1.5 if max(speedups) > 0 else 2)
    ax1.tick_params(axis="y", labelcolor="#C45A3C")

    # 柱顶标加速比
    for bar, s in zip(bars, speedups):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{s:.2f}×", ha="center", fontsize=11, fontweight="bold", color="#C45A3C")

    # Y2: Fused kernel count 折线
    ax2 = ax1.twinx()
    ax2.plot(x, fused_kernels, "o-", color="#2B579A", linewidth=2.5, markersize=10, zorder=5)
    for i, k in enumerate(fused_kernels):
        ax2.annotate(str(k), (x[i], fused_kernels[i]),
                     textcoords="offset points", xytext=(0, 12),
                     fontsize=10, fontweight="bold", color="#2B579A", ha="center")
    ax2.set_ylabel("Fused Kernel Count", fontsize=13, color="#2B579A")
    ax2.set_ylim(0, max(fused_kernels) * 1.5 if max(fused_kernels) > 0 else 5)
    ax2.tick_params(axis="y", labelcolor="#2B579A")

    # 标注论文图 2 融合边界（虚线竖线）
    # 根据设计文档: L3 → L4 是潜在拐点（Softmax 打破 tile-locality）
    ax1.axvline(x=3.5, color="#888888", linestyle="--", linewidth=1.5, alpha=0.8)
    ax1.annotate("Paper Fig.2\nFusion Boundary\n(Softmax / All-Reduce)",
                 xy=(3.5, max(speedups) * 0.85),
                 fontsize=9, color="#555555", ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFDD", edgecolor="#AAAAAA", alpha=0.9))

    # X 轴
    ax1.set_xlabel("Fusion Depth", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(depths, fontsize=12)
    ax1.set_title("图 3: The \"Golden Fusion Point\"\n" +
                  "Speedup (bars) × Kernel Count (line) — where fusion hits physical limits", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", alpha=0.2)

    # 图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="#D4A574", edgecolor="white", label="Speedup (Eager / Fused)"),
        Line2D([0], [0], color="#2B579A", linewidth=2.5, marker="o", markersize=8, label="Fused Kernel Count"),
        Line2D([0], [0], color="#888888", linestyle="--", linewidth=1.5, label="Tile-Locality Boundary"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"图 3 已保存: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="消融实验可视化")
    parser.add_argument("--results", type=str, default=DEFAULT_RESULTS,
                        help="benchmark 结果 JSON 路径")
    parser.add_argument("--profiler", type=str, default=None,
                        help="profiler kernel 计数 JSON 路径（可选）")
    args = parser.parse_args()

    # 加载 benchmark 结果
    if not os.path.exists(args.results):
        print(f"[ERROR] 结果文件不存在: {args.results}")
        print("请先运行 benchmark_ablation.py 生成结果。")
        # 使用模拟数据演示
        print("\n使用模拟数据生成示例图表...")
        results = {
            "L0": {"eager_ms": 0.12, "fused_ms": 0.11, "speedup": 1.09, "reduction_pct": 8.3},
            "L1": {"eager_ms": 0.18, "fused_ms": 0.14, "speedup": 1.29, "reduction_pct": 22.2},
            "L2": {"eager_ms": 0.35, "fused_ms": 0.18, "speedup": 1.94, "reduction_pct": 48.6},
            "L3": {"eager_ms": 0.62, "fused_ms": 0.22, "speedup": 2.82, "reduction_pct": 64.5},
            "L4": {"eager_ms": 0.85, "fused_ms": 0.28, "speedup": 3.04, "reduction_pct": 67.1},
        }
    else:
        results = load_results(args.results)

    # 加载 profiler 数据
    if args.profiler and os.path.exists(args.profiler):
        with open(args.profiler, "r", encoding="utf-8") as f:
            profiler_data = json.load(f)
    else:
        profiler_data = DEFAULT_PROFILER
        print("[INFO] 使用默认 profiler kernel 计数。运行 profiler_verify.py 可获取实测数据。")

    # 画出三张图
    plot_latency_bars(results, os.path.join(OUTPUT_DIR, "fig1_latency_comparison.png"))
    plot_kernel_counts(profiler_data, os.path.join(OUTPUT_DIR, "fig2_kernel_counts.png"))
    plot_golden_fusion_point(results, profiler_data, os.path.join(OUTPUT_DIR, "fig3_golden_fusion_point.png"))

    print("\n所有图表已生成完毕。可截图放入汇报 PPT。")


if __name__ == "__main__":
    main()
