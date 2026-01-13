# -*- coding: utf-8 -*-
"""
主流程入口：批量体积光学仿真与多视角投影自动生成

使用说明：
  1. Normal 模式（原有流程）：
     python main.py --num_configs 200
  
  2. OOD 模式（新增分层数据集）：
     python main.py --generate_mode ood
     所有 split 配置在 config/config.yaml 的 ood_splits 部分
"""

import os
import argparse
import yaml
from datetime import datetime

from src.batch_config_generator import generate_multi_blt_config
from src.batch_simulation_runner import batch_run_mcx_simulations
from src.batch_postprocessor import process_folders
from src.ood_dataset_generator_v3 import generate_ood_splits

DATE_STRING = datetime.now().strftime("%Y%m%d")


def main():
    parser = argparse.ArgumentParser(
        description="体积光学仿真与多视角投影自动生成流程"
    )
    parser.add_argument(
        "--generate_mode",
        type=str,
        choices=["normal", "ood"],
        default="normal",
        help="生成模式：normal（随机样本）或 ood（分层数据集，配置在config.yaml中）",
    )
    parser.add_argument(
        "--num_configs",
        type=int,
        default=2,
        help="normal 模式下生成的样本数，默认为 2",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出目录，默认按当前日期自动生成",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径，默认为 config/config.yaml",
    )

    args = parser.parse_args()

    # 设置输出目录
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = DATE_STRING
        print(f"未指定输出目录，已自动生成: {out_dir}")
    
    os.makedirs(out_dir, exist_ok=True)

    # 加载配置文件
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ========= 根据模式选择流程 =========
    if args.generate_mode == "ood":
        # OOD 数据集生成模式
        print(f"【OOD 模式】生成分层 OOD 数据集...")
        print(f"【OOD】从配置文件读取 split 配置：{args.config}\n")

        # 检查配置
        if "ood_splits" not in config:
            print("【ERROR】配置文件中没有找到 'ood_splits' 配置！")
            print("【ERROR】请在 config.yaml 中添加 ood_splits 部分")
            return

        ood_config = config.get("ood_splits", {})
        
        if not ood_config.get("enabled", False):
            print("【ERROR】OOD 模式未在配置文件中启用！")
            print("【ERROR】请在 config.yaml 的 ood_splits.enabled 设为 true")
            return

        splits_config = ood_config.get("splits", {})
        
        if not splits_config:
            print("【ERROR】配置文件中没有找到任何 split 配置！")
            return

        # 打印配置信息
        print("【OOD】Split 配置:")
        total_samples = 0
        for split_id, split_cfg in splits_config.items():
            num_samples = split_cfg.get("num_samples", 0)
            print(f"  • {split_id}: {num_samples} 个样本 - {split_cfg.get('description', '')}")
            total_samples += num_samples
        print(f"  总计: {total_samples} 个样本\n")

        # 逐个生成每个 split
        for split_id, split_cfg in splits_config.items():
            num_samples = split_cfg.get("num_samples", 100)
            print(f"【OOD】正在生成 {split_id} split (目标: {num_samples} 样本)...")
            
            try:
                generated, rejected = generate_ood_splits(
                    split_id=split_id,
                    num_samples=num_samples,
                    output_dir=out_dir,
                    config=config,
                    split_config=split_cfg,
                    max_rejection_attempts=1000,
                )
                print(f"【OOD】{split_id} 生成完成: {generated} 样本, {rejected} 次拒绝\n")
            except Exception as e:
                print(f"【OOD】{split_id} 生成失败: {e}\n")
                import traceback
                traceback.print_exc()

        print(f"【OOD】所有 split 生成完成，输出目录: {out_dir}")
        
    else:
        # Normal 模式（原有流程）
        print(f"【1】批量生成体积光学仿真配置...（输出目录：{out_dir}）")
        generate_multi_blt_config(
            num_configs=args.num_configs, output_dir=out_dir, config=config
        )
        print("【2】批量运行MCX体积仿真...")
        batch_run_mcx_simulations(out_dir, config=config)
        print("【3】批量仿真输出多角度投影与后处理...")
        process_folders(out_dir, config=config)
        print("流程结束，全部数据及投影结果已生成于:", out_dir)


if __name__ == "__main__":
    main()
