# -*- coding: utf-8 -*-
"""
OOD 数据集生成示例脚本
---------------------
演示如何调用 generate_ood_splits 生成不同 split 的数据集
"""

import sys
import os

# 添加项目路径（根据实际项目结构调整）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ood_dataset_generator import generate_ood_splits, SPLIT_CONFIG
from src.load_config import load_config


def main():
    """
    示例：生成 OOD split 数据集
    """
    # 加载全局配置（需要根据实际项目配置路径调整）
    config = load_config()  # 或自定义加载方式
    
    # 定义所有 split
    splits = [
        ("Train-ID", 200),           # 200 个 ID 训练样本
        ("Test-OOD(B)", 100),        # 100 个尺度比 OOD 样本
        ("Test-OOD(C)", 100),        # 100 个形状数 OOD 样本
        ("Test-OOD(BC)", 100),       # 100 个混合 OOD 样本
    ]
    
    # 输出根目录
    output_base_dir = "./ood_dataset"
    
    print("\n" + "="*60)
    print("OOD 数据集生成工具")
    print("="*60)
    print("\n支持的 Split 配置：")
    for split_id, cfg in SPLIT_CONFIG.items():
        print(f"\n  {split_id}:")
        print(f"    {cfg['description']}")
        print(f"    num_shapes: {cfg['num_shapes']}")
        print(f"    param 范围: [{cfg['min_param']}, {cfg['max_param']}]")
        print(f"    rho 约束: {cfg['rho_constraint']}")
    
    print("\n" + "-"*60)
    print("开始生成数据集...")
    print("-"*60)
    
    # 依次生成各个 split
    total_generated = 0
    total_rejected = 0
    
    for split_id, num_samples in splits:
        print(f"\n[{split_id}] 目标样本数: {num_samples}")
        
        try:
            generated, rejected = generate_ood_splits(
                split_id=split_id,
                num_samples=num_samples,
                output_dir=output_base_dir,
                config=config,
                max_rejection_attempts=1000,
            )
            total_generated += generated
            total_rejected += rejected
        except Exception as e:
            print(f"ERROR in {split_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("生成完成!")
    print(f"  总生成样本数: {total_generated}")
    print(f"  总拒绝次数: {total_rejected}")
    print(f"  输出目录: {output_base_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
