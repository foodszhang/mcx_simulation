from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.batch_config_generator import generate_multi_blt_config
from src.batch_postprocessor import process_folders
from src.batch_simulation_runner import batch_run_mcx_simulations
from src.fem_forward_solver import BatchFemForwardSolver
from src.batch_inverse_runner import BatchInverseRunner


class SimulationBackend(Protocol):
    """统一后端接口：配置生成 -> 仿真 -> 后处理 -> 反演。"""

    name: str

    def generate(self, num_configs: int, output_dir: str, config: dict) -> None:
        ...

    def simulate(self, output_dir: str, config: dict) -> None:
        ...

    def postprocess(self, output_dir: str, config: dict) -> None:
        ...

    def inverse(self, output_dir: str, config: dict) -> None:
        ...


@dataclass(frozen=True)
class PipelineSpec:
    backend: str = "mcx_voxel"


class MCXVoxelBackend:
    """当前默认后端：复用既有体素 MCX 生成逻辑。"""

    name = "mcx_voxel"

    def generate(self, num_configs: int, output_dir: str, config: dict) -> None:
        generate_multi_blt_config(
            num_configs=num_configs,
            output_dir=output_dir,
            config=config,
        )

    def simulate(self, output_dir: str, config: dict) -> None:
        batch_run_mcx_simulations(output_dir, config=config)

    def postprocess(self, output_dir: str, config: dict) -> None:
        process_folders(output_dir, config=config)

    def inverse(self, output_dir: str, config: dict) -> None:
        """MCX 模式反演暂未完全迁移，建议使用 legacy 工具。"""
        # TODO: 集成 legacy inverse 逻辑
        pass


class FEMDiffusionBackend:
    """有限元扩散近似后端：输出节点光通量与采样体场。"""

    name = "fem_de"

    def generate(self, num_configs: int, output_dir: str, config: dict) -> None:
        generate_multi_blt_config(
            num_configs=num_configs,
            output_dir=output_dir,
            config=config,
        )

    def simulate(self, output_dir: str, config: dict) -> None:
        solver = BatchFemForwardSolver(config=config, output_base_dir=output_dir)
        solver.solve_batch()

    def postprocess(self, output_dir: str, config: dict) -> None:
        # FEM 模式也通过统一后处理脚本生成投影图
        # batch_postprocessor.py 会自动检测 fem_volume_fluence.npy 并生成投影
        process_folders(output_dir, config=config)

    def inverse(self, output_dir: str, config: dict) -> None:
        runner = BatchInverseRunner(config=config, output_dir=output_dir)
        runner.run()


class PlaceholderBackend:
    """预留后端：用于未来 MCX 网格流程扩展。"""

    def __init__(self, name: str):
        self.name = name

    def generate(self, num_configs: int, output_dir: str, config: dict) -> None:
        raise NotImplementedError(
            f"后端 '{self.name}' 尚未实现，请先补充 generate/simulate/postprocess 逻辑。"
        )

    def simulate(self, output_dir: str, config: dict) -> None:
        raise NotImplementedError(
            f"后端 '{self.name}' 尚未实现，请先补充 generate/simulate/postprocess 逻辑。"
        )

    def postprocess(self, output_dir: str, config: dict) -> None:
        raise NotImplementedError(
            f"后端 '{self.name}' 尚未实现，请先补充 generate/simulate/postprocess 逻辑。"
        )

    def inverse(self, output_dir: str, config: dict) -> None:
        raise NotImplementedError(
            f"后端 '{self.name}' 尚未实现，请先补充 generate/simulate/postprocess 逻辑。"
        )


def build_backend(backend_name: str) -> SimulationBackend:
    if backend_name == "mcx_voxel":
        return MCXVoxelBackend()
    if backend_name == "fem_de":
        return FEMDiffusionBackend()
    if backend_name == "mcx_mesh":
        return PlaceholderBackend(backend_name)
    raise ValueError(
        "不支持的后端: "
        f"{backend_name}。可选值: mcx_voxel, fem_de, mcx_mesh"
    )


class SimulationPipeline:
    """统一流水线编排：对主入口屏蔽各后端实现细节。"""

    def __init__(self, backend: SimulationBackend):
        self.backend = backend

    def run(self, num_configs: int, output_dir: str, config: dict, logger, run_inverse: bool = False) -> None:
        logger.info(f"当前后端: {self.backend.name}")
        logger.info(f"【1】生成配置 (数量: {num_configs})...")
        self.backend.generate(num_configs=num_configs, output_dir=output_dir, config=config)
        logger.info("✓ 配置生成完成")

        logger.info("【2】运行仿真...")
        self.backend.simulate(output_dir=output_dir, config=config)
        logger.info("✓ 仿真完成")

        logger.info("【3】后处理与结果整理...")
        self.backend.postprocess(output_dir=output_dir, config=config)
        logger.info("✓ 后处理完成")

        if run_inverse:
            logger.info("【4】运行源重建（反演）...")
            self.backend.inverse(output_dir=output_dir, config=config)
            logger.info("✓ 反演完成")
