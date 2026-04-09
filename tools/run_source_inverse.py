from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import load_config
from src.source_inverse import VisiblePointSourceInverse, save_reconstruction_report


def parse_args():
    parser = argparse.ArgumentParser(description="基于 FEM 可见表面 patch partial current 观测的光源反演")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        required=True,
        help="样本目录，例如 /tmp/fem_brain_struct/0",
    )
    parser.add_argument(
        "--sample_id",
        type=int,
        default=0,
        help="样本编号，对应 source-{id}.bin",
    )
    parser.add_argument(
        "--mesh_cache",
        type=str,
        default=None,
        help="mesh 缓存路径，默认自动从样本上级目录 mesh/brain_fem_mesh.npz 推断",
    )
    parser.add_argument(
        "--angles",
        type=int,
        nargs="*",
        default=None,
        help="覆盖配置中的反演角度列表，例如 --angles 0 30 60",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="l1",
        choices=["l1", "legacy_l1", "legacy_tv", "legacy_itcg_vs", "l1_ls", "l2", "laplacian", "tv", "baseline_v1"],
        help="反演方法",
    )
    parser.add_argument(
        "--observation_mode",
        type=str,
        default=None,
        choices=["surface_patch_partial_current", "legacy_surface_nodes", "visible_surface_nodes"],
        help="覆盖配置中的观测模式",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="启用保守 GPU 线性代数路径（仅用于 H 的稠密优化阶段，默认关闭）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="source_inverse_vis",
        help="结果输出目录",
    )
    parser.add_argument(
        "--lambda_val",
        type=float,
        default=None,
        help="覆盖默认的正则化参数 lambda",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=None,
        help="最大迭代次数 (FISTA/L1)",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="禁用 Legacy 流程中的 refinement (debiasing) 步骤",
    )
    parser.add_argument(
        "--excitation-pos",
        nargs=3,
        type=float,
        default=None,
        help="激发光源位置 (x y z)，如果提供则开启激发模式",
    )
    return parser.parse_args()


def infer_mesh_cache(sample_dir: str, config: dict) -> str | None:
    mesh_cache_name = config.get("fem", {}).get("mesh_cache_name", "brain_fem_mesh.npz")
    candidate = Path(sample_dir).parent / "mesh" / mesh_cache_name
    return str(candidate) if candidate.exists() else None


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Ensure source_inverse config dict exists
    inv_conf = config.setdefault("inverse", {}).setdefault("source_inverse", {})

    if args.angles:
        inv_conf["angles"] = args.angles
    
    if args.observation_mode:
        inv_conf["observation_mode"] = args.observation_mode
        
    if args.excitation_pos:
        inv_conf["excitation_pos"] = args.excitation_pos

    if args.gpu:

        config.setdefault("inverse", {}).setdefault("source_inverse", {}).setdefault("gpu", {})[
            "enabled"
        ] = True
    
    if args.lambda_val is not None:
        # Override all lambda types to be safe, or specific ones?
        # Let's set the specific ones based on method, or just all common keys.
        inv_conf = config.setdefault("inverse", {}).setdefault("source_inverse", {})
        inv_conf["l1_lambda"] = args.lambda_val
        inv_conf["legacy_l1_lambda"] = args.lambda_val
        inv_conf["tv_lambda"] = args.lambda_val
        inv_conf["laplacian_lambda"] = args.lambda_val
        inv_conf["l2_lambda"] = args.lambda_val
        inv_conf["reg_lambda"] = args.lambda_val
        
    if args.max_iter is not None:
        inv_conf["l1_max_iter"] = args.max_iter
        inv_conf["legacy_l1_max_iter"] = args.max_iter
        
    if args.no_refine:
        inv_conf["legacy_refine_enabled"] = False
    
    if args.method == "baseline_v1":
        inv_conf["basis_mode"] = "fem_node"
        inv_conf["disable_excitation_scaling"] = True
        eval_conf = inv_conf.setdefault("evaluation", {})
        eval_conf.setdefault("pred_threshold", 0.5)
        eval_conf.setdefault("min_region_size", 10)
        eval_conf.setdefault("cc_connectivity", 26)
        eval_conf.setdefault("cc_dilation_iters", 1)
        eval_conf.setdefault("voxel_spacing", [1.0, 1.0, 1.0])

    mesh_cache = args.mesh_cache or infer_mesh_cache(args.sample_dir, config)
    inverse = VisiblePointSourceInverse(config, sample_dir=args.sample_dir, mesh_cache_path=mesh_cache)
    result = inverse.reconstruct(sample_id=args.sample_id, method=args.method)

    angle_tag = "_".join(str(v) for v in result["angles"])
    out_dir = Path(args.out_dir) / f"sample_{args.sample_id}_{args.method}_{angle_tag}"
    save_reconstruction_report(result, str(out_dir))

    summary_path = out_dir / "inverse_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()
