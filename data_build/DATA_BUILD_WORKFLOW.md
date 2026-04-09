# data_build 旧流程梳理（供后续统一）

本文档基于 `data_build/*.m` 当前代码行为整理，目标是给后续统一进主线提供明确流程定义。

## 1) 当前主入口与真实执行路径

主入口是 `data_build/main.m`。当前活跃路径：

1. `preAmiraMesh` 读入 `.am` 网格并转成节点/四面体/表面三角
2. `Initialization` 生成组织光学参数 `uam, Dm, An`
3. `getsurface` 基于 z 边界保留可见表面三角
4. `FormMatrixF(fyx, sign)`，其中 `fyx=ones`, `sign=ones`
5. `FEMLinear(Dm, uam, ...)` 构造系统矩阵 `Mm`
6. 线性求解 `A = Mm \\ F`
7. 取 `A(index,:)`（index 为表面节点）
8. 构造图拉普拉斯：`getlaplace` + `new_getlaplace_GHB`
9. 保存 `recon_data_*` 与 `recon_info_*`

## 2) 激发相关代码是否“被用到”

有激发相关代码，但按当前 `main.m` 并未调用：

- `FEMLinear_x.m`（使用 `uax, Dx`）
- `sorcesetup.m`
- `get_surface_energy.m`

`main.m` 中只调用了 `FEMLinear`（`Dm, uam`），且 `uax/Dx` 初始化行被注释。  
因此当前默认是**发射侧矩阵准备流程**，不是完整 excitation+emission 两阶段联动。

## 3) 文件角色分组

### A. 网格与几何

- `preAmiraMesh.m`：Amira mesh 解析 + 提取边界三角
- `getsurface.m`：按 z 范围筛表面

### B. FEM 矩阵

- `FEMLinear.m`：主系统矩阵组装（扩散项+吸收项+Robin边界）
- `FormMatrixF.m`：右端矩阵 F 组装
- `FormMatrixM.m`、`FEMLinear_m.m`：与主逻辑高度重复
- `FEMLinear_x.m`：激发分支矩阵（当前未在 main 用）

### C. 标签与样本构造

- `get_signlelabel.m`：单灶标签网格
- `get_double.m`：双灶标签网格
- `get_doublelabel.m`：另一版双灶生成（实现完整性较差）

### D. 正则图与拉普拉斯

- `getlaplace.m`：拓扑邻接归一化拉普拉斯
- `new_getlaplace.m`：多邻域半径拉普拉斯
- `new_getlaplace_GHB.m`：核图归一化拉普拉斯
- `getlaplace_GHB.m`：旧版本，存在未定义变量风险

### E. 可视化导出

- `step3.m`：导出 `results_dat/invivo.dat`
- `tecplot*.m`、`write_dat_part.m`：Tecplot DAT 导出
- `mosetomesh.m`：外部点云映射回网格节点

## 4) 当前流程输入输出

输入：

- `.am` 网格（如 `9029_with_cancer.am`）
- 边界阈值 `min_boundary/max_boundary`

输出：

- `recon_data_*.mat`：`nodes/elementInside/elementSurface/A/...`
- `recon_info_*.mat`：`A/Lap/n_Lap*/...`
- `results_dat/*.dat`（可视化）

## 5) Python 对照实现

已在 `data_build/python_port/` 提供 Python 版本，覆盖主路径所需关键函数，并保留激发相关函数接口（当前不强制参与主流程）。

对应入口：

- `data_build/python_port/main_port.py`

对应说明：

- `data_build/python_port/README.md`

## 6) 后续统一建议（最小化冲突）

1. 以 `main.m` 活跃分支为“基线语义”，先与现有 baseline_v1 对齐
2. 激发分支保留为可选模块，不混入当前 emission-only 主实验
3. 重复 MATLAB 函数（`FormMatrixM/FEMLinear_m/FEMLinear_x`）收敛为单一实现
4. 统一坐标与索引语义（节点编号、1/0-based、surface索引）

