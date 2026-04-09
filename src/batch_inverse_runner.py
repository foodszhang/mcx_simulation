import logging
from pathlib import Path
from src.source_inverse import VisiblePointSourceInverse, save_reconstruction_report

logger = logging.getLogger(__name__)


class BatchInverseRunner:
    """批量反演运行器：遍历输出目录下的所有样本进行反演。"""

    def __init__(self, config: dict, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)

    def run(self) -> None:
        """执行批量反演。"""
        # 获取反演配置
        inverse_conf = self.config.get("inverse", {})
        source_inv_conf = inverse_conf.get("source_inverse", {})
        method = source_inv_conf.get("method", inverse_conf.get("method", "l1"))
        if method == "baseline_v1":
            source_inv_conf["basis_mode"] = "fem_node"
            source_inv_conf["disable_excitation_scaling"] = True
            eval_conf = source_inv_conf.setdefault("evaluation", {})
            eval_conf.setdefault("pred_threshold", 0.5)
            eval_conf.setdefault("min_region_size", 10)
            eval_conf.setdefault("cc_connectivity", 26)
            eval_conf.setdefault("cc_dilation_iters", 1)
            eval_conf.setdefault("voxel_spacing", [1.0, 1.0, 1.0])
        
        # 尝试推断网格缓存路径
        mesh_cache_name = self.config.get("fem", {}).get("mesh_cache_name", "brain_fem_mesh.npz")
        mesh_cache = self.output_dir / "mesh" / mesh_cache_name
        
        if not mesh_cache.exists():
            logger.warning(f"未找到网格缓存文件: {mesh_cache}，尝试重新生成或让 Inverse 类处理...")
            # 如果不存在，传 None 给 VisiblePointSourceInverse，它可能会尝试重新生成或报错
            mesh_cache_path = None
        else:
            mesh_cache_path = str(mesh_cache)

        # 遍历所有数字命名的子目录
        for sample_dir in sorted(self.output_dir.iterdir()):
            if not sample_dir.is_dir() or not sample_dir.name.isdigit():
                continue

            sample_id = int(sample_dir.name)
            logger.info(f"正在为样本 {sample_id} 运行反演 (方法: {method})...")

            try:
                runner = VisiblePointSourceInverse(
                    config=self.config,
                    sample_dir=str(sample_dir),
                    mesh_cache_path=mesh_cache_path
                )
                
                result = runner.reconstruct(sample_id=sample_id, method=method)
                
                # 保存结果到样本目录下的 inverse_results 子目录
                out_subdir = sample_dir / "inverse_results"
                out_subdir.mkdir(exist_ok=True)
                
                save_reconstruction_report(result, str(out_subdir))
                logger.info(f"样本 {sample_id} 反演完成，结果已保存至: {out_subdir}")

            except Exception as e:
                logger.error(f"样本 {sample_id} 反演失败: {e}", exc_info=True)
