# Baseline-v1（FEM, Emission-only）完整流程

本文档固定当前可复现实验流程：**node-based unknown + PGD 非负 Tikhonov**，用于论文1 classical baseline 对比。

## 1. 任务边界（当前版本）

- 仅做 `emission-only`
- 仅做 `matched model + noiseless`
- unknown 固定为 `fem_node`（ROI 内 FEM 节点系数）
- 仅做 `PGD nonnegative Tikhonov`
- 当前单次 `reconstruct(method="baseline_v1")` 仍要求单灶；多灶评估通过批处理脚本方式执行（同一 forward/inverse 语义）


## 2. 代码入口与主链路

- 反演主实现：`src/source_inverse.py`
  - `VisiblePointSourceInverse.reconstruct(..., method="baseline_v1")`
  - `reconstruct_baseline_v1_pgd(...)`
  - `_compute_minrfmt_metrics(...)`
- CLI 入口：`tools/run_source_inverse.py`
- 批量入口：`src/batch_inverse_runner.py`
- FEM 前向与体网格：`src/fem_forward_solver.py` + `src/fem_mesh.py`


## 3. 数学与实现定义

- 优化目标：
  - `min_{x>=0} ||A x - y||_2^2 + reg_lambda ||x||_2^2`
- 变量：
  - `x`：ROI 内 FEM 节点系数（`basis_mode=fem_node`）
  - `A`：观测矩阵（行=measurement，列=ROI 节点）
  - `y`：measurement 向量（由 matched forward 生成）
- PGD 更新：
  - `x0 = 0`
  - `L = reg_lambda + ||A||_2^2`（power iteration 估计谱范数平方）
  - `x_{k+1} = max( x_k - (1/L) * (A^T(Ax_k-y) + reg_lambda*x_k), 0 )`
- 停止条件：
  - 相对更新量 `||x_{k+1}-x_k|| / max(||x_{k+1}||, eps) <= baseline_v1_rel_tol`
  - 或达到 `baseline_v1_max_iter`


## 4. 网格与坐标语义（当前实现）

- 网格来自体素标签，使用 `voxel_tet` 体网格剖分（每个有效体素 cell 固定切分成 6 个四面体）
- 网格构建在 `src/fem_mesh.py::build_tetrahedral_mesh_from_volume`
- `FemDiffusionSolver` 载入已有 mesh cache；若不存在则从体素生成并缓存
- 关键语义：
  - 体素数据在网格构建前按 `XYZ -> ZYX` 对齐
  - 节点物理坐标为 `XYZ(mm)`，体素索引映射按 `lengthunit(dx)` 换算
  - ROI 节点从 `src_range`（体素索引）换算到物理空间后筛选


## 5. 运行方式

## 5.1 单样本（推荐 smoke）

```bash
uv run python tools/run_source_inverse.py \
  --config config/epi_pipeline_pure.yaml \
  --sample_dir true_val_50/0 \
  --sample_id 0 \
  --method baseline_v1 \
  --lambda_val 0.001 \
  --out_dir /tmp/baseline_v1_smoke
```

输出目录中包含：

- `inverse_summary.json`
- `true_source.npy` / `true_source_raw.npy` / `recon_source.npy`
- `metrics_per_sample.csv`
- `summary_by_num_lights.csv`
- `summary_by_num_lights.json`


## 5.2 brain 样本批量指标（含多灶）

1) 用 `config/epi_pipeline.yaml`  
2) `src_range` 需与样本 JSON 的 source pattern 区间一致（brain_val 当前为 `x:[70,104], y:[50,130], z:[86,126]`）  
3) 对每个样本执行 matched forward + baseline_v1 inverse，并统一按 minr_fmt 口径评估

当前最近一次 brain_val(100例)统计结果（`reg_lambda=0.001`）：

- Mean DSC: `0.3803`
- Mean Precision: `0.9951`
- Mean Recall: `0.2407`
- Mean ASSD: `7.6576`
- Mean HD95: `16.0157`
- GT 灶数分布：1灶=24，2灶=48，3灶=28


## 6. 评估口径（与 minr_fmt 对齐）

默认参数：

- `pred_threshold=0.5`
- `min_region_size=10`
- `cc_connectivity=26`
- `cc_dilation_iters=1`
- `voxel_spacing=[1,1,1]`

输出主指标：

- `dice`
- `precision`
- `recall`
- `assd`
- `hd95`

单灶约定：

- `mr/ms/delta_cc` 固定 `NA`（JSON 中为 `null`）


## 7. 已知注意点

- 若 `src_range` 与样本真实 source 区间不一致，会导致评估失真（包含“看起来全满分/全异常”的假象）
- baseline_v1 的 `reconstruct()` 路径默认单灶约束；多灶实验请使用批处理评估脚本方式执行
- 强烈建议固定 `basis_mode=fem_node`、`disable_excitation_scaling=true`


## 8. 最小复现清单

1. 准备 mesh cache（或首次运行自动生成）  
2. 设定 `method=baseline_v1`、`basis_mode=fem_node`、`reg_lambda`  
3. 执行单样本 smoke  
4. 批量跑 `brain_val`/`true_val_50`  
5. 读取 `summary_by_num_lights.csv/json` 汇总入论文表格
