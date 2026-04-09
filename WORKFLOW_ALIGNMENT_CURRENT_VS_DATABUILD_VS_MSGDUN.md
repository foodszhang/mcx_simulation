# 当前流程 / data_build / MS_GDUN 对齐文档

本文档用于统一三套流程的语义、数据结构和后续整合路径：

- 当前主线：`mcx_simulation` 中 baseline-v1（FEM, emission-only）
- 历史流程：`data_build`（MATLAB 传统 FEM 数据构建）
- 网络训练：`/home/foods/pro/MS_GDUN_for_MICCAI2026`

---

## 1. 三套流程的定位

## 1.1 当前主线（baseline-v1）

- 目标：论文1 classical baseline（对比用）
- 数学：`min_{x>=0} ||Ax-y||^2 + reg_lambda ||x||^2`
- unknown：FEM ROI 节点系数（`fem_node`）
- forward/inverse：matched + noiseless + emission-only
- 评估：minr_fmt 对齐口径（Dice/ASSD/HD95/Precision/Recall）

入口：

- `src/source_inverse.py`
- `tools/run_source_inverse.py`
- `src/batch_inverse_runner.py`

---

## 1.2 data_build（MATLAB）

- 目标：构建传统 FEM 线性系统和训练/重建相关中间数据
- 当前主入口：`data_build/main.m`
- 当前主路径：`A = Mm \\ F` + 拉普拉斯 + 保存 mat

关键产物（按脚本定义）：

- `recon_data_*.mat`（`A`, `nodes`, `elementInside`, `elementSurface`...）
- `recon_info_*.mat`（`A`, `Lap`, `n_Lap*`...）
- 标签相关：`Label_Data`, `test_b`（由 `get_signlelabel/get_double` 生成）

---

## 1.3 MS_GDUN（网络）

路径：

- `/home/foods/pro/MS_GDUN_for_MICCAI2026`

训练读入字段（`model/Dataset.py`）：

- `A`
- `Lap`
- `Lap0` 或 `n_Lap0`
- `n_Lap1`, `n_Lap2`, `n_Lap3`
- `test_b`
- `Label_Data`
- `nodes`
- （可选 GCN pad）`S_global_indices`

训练脚本使用：

- `train.py` 默认 `data/train_data_single_9029_new.mat`

---

## 2. data_build 是否“实际使用激发流程”

结论：**代码存在激发分支，但 `main.m` 当前默认路径未使用。**

### 2.1 存在但未被 main 调用

- `FEMLinear_x.m`（`uax/Dx`）
- `sorcesetup.m`
- `get_surface_energy.m`

### 2.2 main.m 当前实际调用

- `preAmiraMesh`
- `Initialization`（只取 `uam, Dm, An`）
- `getsurface`
- `FormMatrixF`
- `FEMLinear`（发射侧参数 `Dm/uam`）
- `getlaplace`, `new_getlaplace_GHB`

因此 data_build 当前主链路是“发射侧线性系统构建”，不是完整 excitation+emission 两阶段求解。

---

## 3. 字段级对齐（核心）

| 语义 | data_build 侧 | MS_GDUN 侧 | 当前 baseline-v1 侧 |
|---|---|---|---|
| 系统矩阵 | `A` | `A` | `operator` |
| 基础拉普拉斯 | `Lap` | `Lap` | 可由 mesh 图再构建 |
| 多尺度图 | `n_Lap0..3` | `Lap0/n_Lap0`, `n_Lap1..3` | 当前未作为 solver 输入 |
| 观测 | `test_b` | `test_b` | `measurement` |
| 标签 | `Label_Data` | `Label_Data` | `true_source_eval / coeff` |
| 节点坐标 | `nodes` | `nodes` | `solver.mesh.nodes` |
| 表面索引 | `S_global_indices`（另脚本产） | `S_global_indices`（可选） | `row_slices_by_angle` 等观测映射 |

要点：

- MS_GDUN 的输入字段与 data_build 产物高度同构。
- baseline-v1 与 MS_GDUN 在“线性反演 vs 学习反演”层面不同，但底层 `A/graph/nodes` 可共享。

---

## 4. 数据流对齐图（文字版）

1. 网格准备  
`Amira(.am)` / 体素网格 -> `nodes`, `elementInside`, `elementSurface`

2. 物理矩阵构建  
`FEMLinear + FormMatrixF` -> `A`

3. 先验图构建  
`getlaplace/new_getlaplace_GHB` -> `Lap`, `n_Lap*`

4. 样本标签构建  
`get_signlelabel/get_double` -> `Label_Data`, `test_b`

5. 数据打包  
写入 `train_data_*.mat`（供 MS_GDUN）或 baseline 运行目录（供传统反演）

---

## 5. 当前可复用与不一致点

## 5.1 可复用

- `A` 的定义思路（线性观测算子）
- 图拉普拉斯多尺度先验
- 节点级标签生成逻辑（单灶/双灶）

## 5.2 不一致

- 当前 baseline-v1 固定 emission-only + PGD 非负 Tikhonov
- data_build 中不少脚本是全局变量风格 + 强硬编码文件名（`9029*`）
- MS_GDUN 数据是“二次打包后的 mat”，不是直接读取 `recon_info_*.mat`

---

## 6. 已完成的 Python 对照迁移

目录：`data_build/python_port/`

已迁移模块：

- `initialization.py`
- `mesh_io.py`
- `fem_matrices.py`
- `laplacian.py`
- `labels.py`
- `source_ops.py`
- `tecplot_export.py`
- `main_port.py`

用途：

- 保留 data_build 语义，便于后续统一进 Python 主线
- 为将来构建 “data_build -> MS_GDUN 训练mat” 一键脚本打基础

---

## 7. 统一建议（后续执行顺序）

1. 固定统一数据契约（字段名、shape、坐标语义）  
2. 先做打包器：`A/Lap/n_Lap*/Label_Data/test_b/nodes -> train_data_*.mat`  
3. 用同一 mesh/同一 A 对比 baseline-v1 与 MS_GDUN  
4. 再决定是否引入激发分支（默认先不并入主实验）

---

## 8. 一句话结论

- `data_build` 确实有数字鼠模型与初始流程。  
- `MS_GDUN` 与这套数据结构是**对得上的**（字段层面高度匹配）。  
- 但当前 data_build 主入口并未启用激发分支；默认是发射侧线性系统构建流程。

