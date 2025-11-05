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
    print(f"【1】批量生成体积光学仿真配置……（输出目录：{out_dir}）")
    generate_multi_blt_config(num_configs=num_configs, output_dir=out_dir)
    print("【2】批量运行MCX体积仿真……")
    batch_run_mcx_simulations(out_dir)
    print("【3】批量仿真输出多角度投影与后处理……")
    process_folders(out_dir)
    print("流程结束，全部数据及投影结果已生成于:", out_dir)


if __name__ == "__main__":
    main()
