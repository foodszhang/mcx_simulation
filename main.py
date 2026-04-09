# -*- coding: utf-8 -*-
"""
主流程入口：批量体积光学仿真与多视角投影自动生成
本文件为顶层任务调度，可直接使用 uv run python main.py 执行。
所有输出、流程及日志均为中文，遵循医学/生物领域最佳实践。

支持两种生成模式：
  - normal: 原有的标准流程（配置生成 -> MCX仿真 -> 后处理）
"""

import os
import sys
import argparse
import logging
from datetime import datetime

try:
    import yaml
except ImportError:
    sys.exit("错误: 缺少 PyYAML 依赖，请运行: pip install pyyaml")

from src.pipeline import build_backend, SimulationPipeline

# 初始化日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATE_STRING = datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="批量体积光学仿真与多视角投影自动生成流程（全流程中文，深度学习与医学最佳实践）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 标准流程（默认）
  python main.py --num_configs 5
  
  # 指定输出目录和配置文件
  python main.py --out_dir ./results --config config/custom.yaml
  
  # 详细日志输出
  python main.py --verbose
        """,
    )
    parser.add_argument(
        "--num_configs", type=int, default=2, help="批量生成仿真配置的数量 (默认: 2)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="仿真输出目录，默认按当前日期时间自动命名",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径 (默认: config/config.yaml)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["mcx_voxel", "fem_de", "mcx_mesh"],
        help="仿真后端选择，未指定时优先使用配置文件 pipeline.backend 或默认 mcx_voxel",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="先执行体素/材料初始化（tools/main_volume_and_material_config.py）再进入主流程",
    )
    parser.add_argument(
        "--bootstrap_only",
        action="store_true",
        help="仅执行体素/材料初始化并退出，不运行仿真与后处理",
    )
    parser.add_argument(
        "--run_inverse",
        action="store_true",
        help="在仿真与后处理后，自动运行源重建（反演）",
    )
    parser.add_argument("--verbose", action="store_true", help="启用详细日志输出")

    return parser.parse_args()


def load_config(config_path):
    """加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        dict: 配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML解析错误
        ValueError: 配置文件为空
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError(f"配置文件为空: {config_path}")
            return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML解析错误: {e}")


def setup_output_dir(out_dir):
    """设置输出目录

    Args:
        out_dir: 指定的输出目录，如果为None则自动生成

    Returns:
        str: 最终的输出目录路径
    """
    if not out_dir:
        out_dir = f"output_{DATE_STRING}"
        logger.info(f"未指定输出目录，已自动生成: {out_dir}")

    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"输出目录已就绪: {out_dir}")

    return out_dir


def should_run_bootstrap(config):
    """若缺少关键输入产物则自动触发初始化。"""
    generated_bin = config.get("generated_bin_path")
    generated_mat = config.get("generated_material_yaml_path")
    return not (
        generated_bin
        and generated_mat
        and os.path.exists(generated_bin)
        and os.path.exists(generated_mat)
    )


def run_bootstrap(config_path):
    """调用初始化脚本生成体素与材料，并回写配置。"""
    from tools.main_volume_and_material_config import bootstrap_volume_and_material

    logger.info("执行初始化：生成体素与材料配置...")
    target_config = bootstrap_volume_and_material(config_path=config_path)
    logger.info(f"✓ 初始化完成，配置已更新: {target_config}")


def run_normal_pipeline(args, config):
    """运行标准流程

    Args:
        args: 命令行参数
        config: 配置字典

    Raises:
        Exception: 流程执行失败
    """
    out_dir = args.out_dir

    try:
        backend_name = args.backend or config.get("pipeline", {}).get(
            "backend", "mcx_voxel"
        )
        backend = build_backend(backend_name)
        pipeline = SimulationPipeline(backend=backend)
        pipeline.run(
            num_configs=args.num_configs,
            output_dir=out_dir,
            config=config,
            logger=logger,
            run_inverse=args.run_inverse,
        )

        logger.info(f"\n✓ 流程结束，全部数据及投影结果已生成于: {out_dir}")

    except Exception as e:
        logger.error(f"标准流程执行失败: {e}", exc_info=args.verbose)
        raise


def main():
    """主流程入口"""
    try:
        # 解析命令行参数
        args = parse_arguments()

        # 设置日志级别
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("已启用详细日志输出")

        logger.info("=" * 70)
        logger.info("MCX 体积光学仿真流程启动")
        logger.info(f"配置文件: {args.config}")
        logger.info(f"后端参数: {args.backend or '(自动)'}")
        logger.info(f"初始化参数: bootstrap={args.bootstrap}, bootstrap_only={args.bootstrap_only}")
        logger.info("=" * 70)

        # 加载配置文件
        config = load_config(args.config)
        logger.debug(f"配置已加载: {len(config)} 个配置项")

        if args.bootstrap or args.bootstrap_only or should_run_bootstrap(config):
            if not args.bootstrap and not args.bootstrap_only:
                logger.info("检测到缺失初始化产物，自动执行bootstrap。")
            run_bootstrap(args.config)
            config = load_config(args.config)

        if args.bootstrap_only:
            logger.info("bootstrap_only=True，初始化完成后退出。")
            return

        # 设置输出目录
        args.out_dir = setup_output_dir(args.out_dir)

        # 根据生成模式选择流程
        run_normal_pipeline(args, config)

        logger.info("=" * 70)
        logger.info("✓ 流程执行成功")
        logger.info("=" * 70)

    except FileNotFoundError as e:
        logger.error(f"文件不存在: {e}")
        sys.exit(1)
    except (yaml.YAMLError, ValueError) as e:
        logger.error(f"配置文件错误: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"流程执行出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
