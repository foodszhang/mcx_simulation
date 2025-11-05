# 项目交接文档 / HANDOVER DOCUMENT

---

## 1. 项目简介
本项目为基于Python实现的医学/生物成像模拟及数据处理工具集，包含体素、投影、仿真、配置自动生成、3D可视化和批量数据处理等功能，适用于红外光成像/荧光成像等生物物理场景的数据回放、算法开发与实验模拟。核心技术栈包括pytorch、numpy等，鼓励严格遵循深度学习和科学计算最佳实践。**所有开发、维护与文档一律采用中文**。

---

## 2. 目录结构与主要脚本功能

### 2.1 主流程&入口模块
- `main.py`：全局主入口文件，调度src目录下的仿真、批量生成、投影后处理各模块，实现批量体积光学仿真与多视角投影的完整主流程。
- `tests/test.py`：用于单元测试/调试各数据及配置读取，辅助开发。

### 2.2 体素与数据预处理
- `preprocessing.py`：体素网格分块、预处理工具，支持大体积数据分区存储与后处理。
- `FMTData_human.py`：主要类`NIRIIFMTDataGenerator`负责医学成像数据（荧光、激发等）仿真，包括体积生成、探测器位置计算、噪声和数据标准化、批量样本生成。

### 2.3 配置生成与参数自动化
- `gen_config.py`/`green_gen.py`/`make_green_mat.py`/`batch_config_generator.py`/`batch_postprocessor.py`/`batch_simulation_runner.py`：批量生成仿真或配置信息、格林函数参数、批量合成数据与配置等。

### 2.4 三维图像及投影
- `vis_3d.py`：3D医学图像（如分割体素/点云）可视化及样本数据生成。
- `show_proj.py`/`show_projection.py`：投影/热图矩阵可视化工具，支持批量/单张多视角显示。
- `get_simple_projection.py`/`gen_mul_projection.py`：体积数据多方向投影、特征变换等仿真工具，`VolumeProjector`等类负责多视角投影分析及显示。
- `preview.py`：配置/投影的快速预览与调试脚本。

### 2.5 绿色材料相关
- `fix_green.py`、`randomize_media.py`、`save_green_mat.py`等负责绿色材料体素/仿真数据修正与扩充。

### 2.6 基础“工具函数”
- `simple_gen.py`: 常用于三维旋转、形状生成、体积/材料合成。
- `load_jnii.py`：用于加载特定格式体素/图像数据，支持滑动交互。

### 2.7 其他数据结构
- `*.json`文件：如`test.json`、`volume.json`、`why.json`，定义仿真/测试/体积基本参数与样本。

---

## 3. 依赖管理规范
1. 项目建议使用`uv`工具进行环境和包管理，**严禁直接用pip**。
2. 安装依赖请统一用如`uv add numpy torch`指令，自动写入`pyproject.toml`。
3. 运行脚本务必用`uv run python xxx.py`，避免环境污染与兼容性问题。

---

## 4. 数据流与典型工作流
### 4.1 批量体积光学仿真与多视角投影自动生成流程
该工作流横跨batch_config_generator.py、batch_simulation_runner.py、batch_postprocessor.py等脚本，执行流程概述如下：
1. 通过`gen_test_blt_config.py`批量自动生成大规模医学成像/光学仿真配置文件，每条配置含介质分层、多参数扰动与noise策略。
2. 用`gen_test_blt_all_data.py`批量遍历所有配置，自动调用MCX完成倒排体积光学仿真并批量存储结果，保障仿真可复现。
3. 利用`gen_test_blt_other_data.py`对仿真输出数据进行多角度投影、深度特征矩阵等后处理，适配各类下游模型及可视化分析。

### 4.2 其他流程纲要
- 可使用`gen_config.py`、`green_gen.py`等自动生成仿真参数文件和材料参数。
- 用`FMTData_human.py`、`preprocessing.py`对原始体积数据和医学图像批处理与分块。
- 其他主流程可由`main.py`串联，实现个性化仿真与批量结果输出。
- 投影和可视化推荐用`show_proj.py`、`vis_3d.py`，支持医学图像多角度分析。

---

## 5. 常见数据格式说明
- `*.json`：通用配置/元数据格式，例如体素、投影参数。
- `*.npz`、`*.npy`：批量医学图像与体素数据矩阵等。
- `*.mat`、`*.jnii`等：特定格式体积/材料/投影仿真输出；如jnii为自定义体素（3D）序列。

---

## 6. 最佳实践与代码规范
- 必须充分注释主流程与算法核心代码。
- 新增模块需补交接文档与典型用例。
- 用type hint与断言保证类型安全。
- 保证Pytorch与Numpy等Tensor操作规范，防止显存/内存泄漏。
- 医学/生物模拟过程建议遵循数据隐私与科研复现最佳实践，敏感数据隔离。
- 推荐开发完成后运行核心脚本的功能测试，确保算法可用。

---

## 7. 环境与运行示例
```bash
uv add numpy torch matplotlib
uv run python main.py
uv run python FMTData_human.py
uv run python preprocessing.py
uv run python vis_3d.py
```
根据业务需要组合调用上文描述的各脚本及json数据。

---

## 8. 常见问题与维护建议
- **环境冲突**：请使用`uv`保持依赖隔离，遇到兼容性需及时升级pyproject.toml。
- **数据格式不对**：严格用工具脚本生成/校验配置与体素等文件，避免手写干扰。
- **性能瓶颈**：大数据量建议batch/多进程，避免单线程执行。
- **调试建议**：善用test.py与可视化脚本辅助查错。

---

## 9. 交接与后续开发建议
- 建议后续开发严格按此文档主流程与功能模块划分。
- 重要变更、优化务必补齐注释及flow文档。
- 研发/测试数据请及时归档管理。

---

如有疑问请参阅history commit或联系原开发人员。祝开发顺利！

