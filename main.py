# -*- coding: utf-8 -*-
"""
主流程入口：批量体积光学仿真与多视角投影自动生成
本文件为顶层任务调度，可直接使用uv run python main.py 执行。
所有输出、流程及日志均为中文，遵循医学/生物领域最佳实践。
"""
import os
import argparse

from src.batch_config_generator import generate_multi_blt_config
from src.batch_simulation_runner import batch_run_mcx_simulations
from src.batch_postprocessor import process_folders
from datetime import datetime

DATE_STRING = datetime.now().strftime("%Y%m%d")


def main():
    parser = argparse.ArgumentParser(
        description="批量体积光学仿真与多视角投影自动生成流程（全流程中文，深度学习与医学最佳实践）"
    )
    parser.add_argument(
        "--num_configs", type=int, default=2, help="批量生成仿真配置的数量，默认为2"
    )
    parser.add_argument(
        "--out_dir", type=str, default=None, help="仿真输出目录，默认按当前日期自动命名"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="配置文件路径，默认为config/config.yaml"
    )  # 新增：支持动态指定配置文件路径，便于多数据集灵活切换
    args = parser.parse_args()

    num_configs = args.num_configs
    # 若用户未指定 out_dir，则自动生成带日期的目录并传递全流程
    if args.out_dir:
        out_dir = args.out_dir
    else:
        from datetime import datetime

        out_dir = DATE_STRING
        print(f"未指定输出目录，已自动生成: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    # ========= 中文流程：唯一加载配置文件，并全流程穿透 config 参数 =========
    # 首先根据--config参数动态加载全局配置文件
    import yaml
    config_path = args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"【1】批量生成体积光学仿真配置……（输出目录：{out_dir}）")
    # 全流程将 config 明确传入每个主流程模块，确保参数驱动
    generate_multi_blt_config(num_configs=num_configs, output_dir=out_dir, config=config)  # 调用批量配置生成
    print("【2】批量运行MCX体积仿真……")
    batch_run_mcx_simulations(out_dir, config=config)
    print("【3】批量仿真输出多角度投影与后处理……")
    process_folders(out_dir, config=config)
    print("流程结束，全部数据及投影结果已生成于:", out_dir)


if __name__ == "__main__":
    main()
