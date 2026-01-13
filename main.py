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

from src.batch_config_generator import generate_multi_blt_config
from src.batch_simulation_runner import batch_run_mcx_simulations
from src.batch_postprocessor import process_folders

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
        logger.info(f"【1】批量生成体积光学仿真配置 (数量: {args.num_configs})...")
        generate_multi_blt_config(
            num_configs=args.num_configs, output_dir=out_dir, config=config
        )
        logger.info("✓ 配置生成完成")

        logger.info("【2】批量运行MCX体积仿真...")
        batch_run_mcx_simulations(out_dir, config=config)
        logger.info("✓ 仿真完成")

        logger.info("【3】批量仿真输出多角度投影与后处理...")
        process_folders(out_dir, config=config)
        logger.info("✓ 后处理完成")

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
        logger.info("=" * 70)

        # 加载配置文件
        config = load_config(args.config)
        logger.debug(f"配置已加载: {len(config)} 个配置项")

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
