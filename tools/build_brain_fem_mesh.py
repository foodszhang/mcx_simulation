#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.fem_forward_solver import FemDiffusionSolver
from src.load_config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="根据brain体素数据生成粗FEM四面体网格")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="mesh缓存输出路径，默认写入 output_dir/mesh/brain_fem_mesh.npz",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    mesh_cache = args.out
    if mesh_cache is None:
        mesh_cache = os.path.join(
            config.get("output_dir", "./output_data"),
            "mesh",
            config.get("fem", {}).get("mesh_cache_name", "brain_fem_mesh.npz"),
        )
    solver = FemDiffusionSolver(
        volume_path=config["generated_bin_path"],
        media_yaml_path=config["generated_material_yaml_path"],
        volume_shape=tuple(config["volume_shape"]),
        dx=config.get("lengthunit", 0.1),
        mesh_cache_path=mesh_cache,
        fem_config=config.get("fem", {}),
    )
    print(f"mesh已生成: {mesh_cache}")
    print(f"节点数: {len(solver.mesh.nodes)}")
    print(f"单元数: {len(solver.mesh.elements)}")


if __name__ == "__main__":
    main()
