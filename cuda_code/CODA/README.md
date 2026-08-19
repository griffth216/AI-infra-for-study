# CODA — Epilogue Fusion Ablation Experiment

消融实验：量化算子融合深度与性能收益的关系。

## 文件说明

| 文件 | 用途 |
|------|------|
| `benchmark_ablation.py` | 主实验脚本：L0-L4 各深度的 eager vs fused 延迟对比 |
| `profiler_verify.py` | Profiler 验证：统计 kernel launch 数量，导出 chrome trace |
| `plot_results.py` | 可视化：生成 3 张图表（延迟柱状图、kernel 数对比、黄金融合点） |

## 运行方式

```bash
# 1. 运行 benchmark（约 10-15 分钟）
cd cuda_code/CODA
python benchmark_ablation.py

# 2. Profiler 验证（约 5 分钟）
python profiler_verify.py

# 3. 画图
python plot_results.py
```

## 实验设计

对应 `notes/week4/消融实验设计.md`。

5 个融合深度 (L0-L4)，每个深度同时跑两版：

| 版本 | 实现 | 含义 |
|------|------|------|
| Eager (未融合) | PyTorch eager mode | 每个 op 独立 launch kernel → 多次 HBM 往返 |
| Fused (融合) | `torch.jit.script` (TorchScript) | TensorExpr fuser 自动融合 element-wise op → 减少 kernel launch |

**融合后端说明**：`torch.compile(backend="inductor")` 在 CUDA 上依赖 Triton，而 Triton 目前不支持 Windows 原生运行。本实验使用 TorchScript 的 TensorExpr fuser 作为替代方案，其融合机制与 inductor 类似（均捕获计算图并将连续 element-wise op 合并为单个 CUDA kernel）。

## 关键发现

### Kernel Launch 数量对比 (L2 示例)

```
Eager L2: 9 个 CUDA kernel
  - vectorized_elementwise_kernel (add)    360 us
  - PowKernel                              261 us
  - reduce_kernel (mean)                   105 us
  - sqrt_kernel                              1 us
  - reciprocal_kernel                        1 us
  - elementwise_kernel (mul × rstd)          1 us
  - elementwise_kernel (mul × gamma) ×2    585 us
  + cuBLAS GEMM                          6005 us

Fused L2: 4 个 CUDA kernel
  - fused_add_pow                         518 us  ← add + pow 融合
  - fused_add_sqrt_reciprocal_mul_mul_mul  353 us  ← add+sqrt+rsqrt+mul+mul 融合
  - reduce_kernel (mean)                  184 us
  + cuBLAS GEMM                          6008 us

→ 减少 56% kernel launch
```

### 延迟对比 (RTX 4060 Laptop, BF16)

| Depth | Eager (ms) | Fused (ms) | Speedup |
|-------|-----------|-----------|---------|
| L0 | 4.75 | 4.77 | 0.99× |
| L1 | 5.10 | 5.03 | 1.01× |
| L2 | 5.99 | 5.65 | 1.06× |
| L3 | 15.92 | 14.79 | 1.08× |
| L4 | 17.76 | 16.45 | 1.08× |

加速比从 L0→L4 从 1.0× 增长到 1.08×，L3→L4 增速放缓 → **黄金融合点**信号出现。
