# -*- coding: utf-8 -*-
"""
OOD 数据集生成模块（与完整仿真流程集成）- 配置文件版本
-------------------------------------------
为论文 B（尺度比 rho）+ C（shape 数）OOD 评估构造受控数据集
支持在线筛选/重采样，确保 split 约束严格性
与 MCX 仿真、后处理流程完全集成
所有 split 配置从 config.yaml 中读取
"""

import json
import os
from datetime import datetime
import random
import shutil
from copy import deepcopy
import yaml
from src.shape_generator import generate_multiple_shapes
from src.batch_simulation_runner import batch_run_mcx_simulations
from src.batch_postprocessor import process_folders
import numpy as np


def compute_rho(shapes_info):
    """计算样本级尺度比 rho"""
    if not shapes_info or len(shapes_info) == 0:
        return 1.0

    r_eq_list = []
    for shape in shapes_info:
        shape_type = shape.get("type", "sphere")
        param = shape.get("param")

        if shape_type == "sphere":
            r_eq = param
        elif shape_type == "ellipsoid":
            rx, ry, rz = param
            r_eq = (rx * ry * rz) ** (1 / 3)
        elif shape_type == "cube":
            r_eq = param / (2 ** (1 / 3))
        elif shape_type == "cylinder":
            radius, height = param
            r_eq = (radius**2 * height) ** (1 / 3)
        else:
            r_eq = param if isinstance(param, (int, float)) else param[0]

        r_eq_list.append(r_eq)

    if len(r_eq_list) == 1:
        return 1.0

    rho = max(r_eq_list) / min(r_eq_list)
    return rho


def check_rho_constraint(rho, constraint):
    """检查 rho 是否满足约束"""
    min_rho = constraint[0]
    max_rho = constraint[1] if len(constraint) > 1 else float("inf")

    # 处理字符串 'inf' 的情况
    if isinstance(max_rho, str) and max_rho == "inf":
        max_rho = float("inf")

    return min_rho <= rho <= max_rho


def generate_ood_splits(
    split_id="Train-ID",
    num_samples=200,
    output_dir=None,
    config=None,
    split_config=None,
    max_rejection_attempts=1000,
    run_simulation=True,
):
    """
    为 OOD 评估生成分层数据集：支持在线筛选/重采样确保 split 严格性

    :param split_id: split 名称 ("Train-ID", "Test-OOD(B)", "Test-OOD(C)", "Test-OOD(BC)")
    :param num_samples: 目标样本数
    :param output_dir: 输出根目录（所有 split 共享根目录）
    :param config: 全局配置参数
    :param split_config: 该 split 的具体配置字典（从 config.yaml 中读取）
    :param max_rejection_attempts: 单个样本最大重采样次数
    :param run_simulation: 是否运行 MCX 仿真和后处理（True 则完整流程，False 只生成配置）
    :return: (generated_count, rejected_count)
    """
    assert config is not None, "必须传入 config"
    assert split_config is not None, f"必须为 split_id '{split_id}' 传入 split_config"

    if output_dir is None:
        date_string = datetime.now().strftime("%Y%m%d")
        output_dir = f"./{date_string}_OOD"

    # 从 split_config 中读取配置
    split_cfg = split_config

    # ===== 重要：所有 split 共享根目录，每个 split 在根目录下创建子目录 =====
    os.makedirs(output_dir, exist_ok=True)
    split_dir = os.path.join(output_dir, split_id)
    os.makedirs(split_dir, exist_ok=True)

    # 确保每个 split 目录下有体素 bin（文件名由 config['generated_bin_path'] 决定）
    volume_src_path = config["generated_bin_path"]
    volume_dst_path = os.path.join(split_dir, os.path.basename(volume_src_path))
    if not os.path.exists(volume_dst_path):
        shutil.copy(volume_src_path, volume_dst_path)

    # 加载配置
    with open(config["generated_material_yaml_path"], "r", encoding="utf-8") as yf:
        media_list = yaml.safe_load(yf)

    volume_shape = config["volume_shape"]
    # 保持后续 volume_file 路径为 split 目录的上一层
    volume_file = "../" + os.path.basename(volume_dst_path)



    src_range_z = config["src_range"]["z"]
    src_range_y = config["src_range"]["y"]
    src_range_x = config["src_range"]["x"]
    max_rotation = config["src_range"].get("max_rotation", 0)

    src_voxel_size = (
        src_range_x[1] - src_range_x[0],
        src_range_y[1] - src_range_y[0],
        src_range_z[1] - src_range_z[0],
    )

    file_naming = config.get(
        "file_naming",
        {
            "normal": {"config": "{id}.json", "result": "{id}.jnii"},
            "noscatter": {"config": "no_{id}.json", "result": "no_{id}.jnii"},
        },
    )

    def format_filename(template, _id):
        return template.format(id=_id)

    generated_count = 0
    rejected_count = 0
    metadata_list = []

    # 获取 split 配置中的参数
    num_shapes_list = split_cfg.get("num_shapes", [1, 2])
    min_param = split_cfg.get("min_param", 3)
    max_param = split_cfg.get("max_param", 10)
    rho_constraint_raw = split_cfg.get("rho_constraint", [0, float("inf")])

    # 处理配置中可能的 'inf' 字符串
    rho_constraint = []
    for val in rho_constraint_raw:
        if isinstance(val, str) and val == "inf":
            rho_constraint.append(float("inf"))
        else:
            rho_constraint.append(
                float(val) if isinstance(val, (int, float, str)) else val
            )
    rho_constraint = tuple(rho_constraint)

    print(f"\n========== 生成 {split_id} (目标: {num_samples} 样本) ==========")
    print(
        f"配置: num_shapes={num_shapes_list}, "
        f"min_param={min_param}, "
        f"max_param={max_param}, "
        f"rho_constraint={rho_constraint}"
    )
    print(f"输出目录: {split_dir}")

    sample_id = 0
    while generated_count < num_samples:
        rejection_count = 0
        success = False
        target_scales = None

        # 条件采样：如有 rho_constraint 且 num_shapes > 1，先采目标尺度集合
        while rejection_count < max_rejection_attempts:
            num_shapes = random.choice(num_shapes_list)
            use_conditional = (
                rho_constraint is not None and len(rho_constraint) == 2 and num_shapes > 1
            )
            if use_conditional:
                # ----------- 条件采样 target_scales -----------
                s_min = min_param
                s_max = max_param
                # 1. small
                s_small = random.randint(s_min, min(s_min + 1, s_max))
                # 2. rho_target
                rho_min = rho_constraint[0]
                rho_max = rho_constraint[1] if rho_constraint[1] != float('inf') else rho_min + 0.7
                rho_target = random.uniform(rho_min, min(rho_min + 0.7, rho_max))
                # 3. large
                s_large = int(round(rho_target * s_small))
                s_large = max(s_min, min(s_large, s_max))
                if num_shapes == 2:
                    target_scales = [s_small, s_large]
                elif num_shapes == 3:
                    # 均匀采中间尺度
                    s_mid = random.randint(s_small, s_large)
                    target_scales = [s_small, s_mid, s_large]
                else:
                    # >3: small/large，中间均匀采
                    mids = [int(round(s_small + (s_large - s_small) * i / (num_shapes - 1))) for i in range(1, num_shapes - 1)]
                    target_scales = [s_small] + mids + [s_large]
            else:
                target_scales = None

            source_arr, shapes_info = generate_multiple_shapes(
                src_voxel_size,
                num_shapes,
                min_param=min_param,
                max_param=max_param,
                max_rotation=max_rotation,
                shape_types=file_naming.get("shape_types", ["sphere", "ellipsoid"]),
                target_scales=target_scales,
            )

            # 计算 rho
            rho = compute_rho(shapes_info)

            # 检查约束
            if check_rho_constraint(rho, rho_constraint):
                success = True
                break
            else:
                # 优先局部调整（如重采 small/large），尝试微调 target_scales
                if use_conditional and target_scales is not None:
                    # 只调整 large 或 small
                    adjust_idx = random.choice([0, -1])
                    if adjust_idx == 0:
                        # 重采 small
                        s_small = random.randint(s_min, min(s_min + 1, s_max))
                        s_large = target_scales[-1]
                        rho_target = s_large / s_small
                    else:
                        # 重采 large
                        s_large = random.randint(max(s_small + 1, s_min), s_max)
                        s_small = target_scales[0]
                        rho_target = s_large / s_small
                    if num_shapes == 2:
                        target_scales = [s_small, s_large]
                    elif num_shapes == 3:
                        s_mid = random.randint(s_small, s_large)
                        target_scales = [s_small, s_mid, s_large]
                    else:
                        mids = [int(round(s_small + (s_large - s_small) * i / (num_shapes - 1))) for i in range(1, num_shapes - 1)]
                        target_scales = [s_small] + mids + [s_large]
            rejection_count += 1

        if not success:
            rejected_count += 1
            print(
                f"  警告: 样本 {sample_id} 达到最大重采样次数({max_rejection_attempts})，跳过"
            )
            continue


        # ===== 保存该样本到 split_dir 下的独立子目录 =====
        session_id = f"{generated_count:06d}"
        config_subdir = os.path.join(split_dir, session_id)
        os.makedirs(config_subdir, exist_ok=True)

        source_filename = f"source-{session_id}.bin"
        full_source_path = os.path.join(config_subdir, source_filename)

        # 保存体素数据（保持原有的物理仿真流程）
        source_arr = source_arr.astype(np.float32)
        source_arr.tofile(full_source_path)

        # 构建 metadata
        metadata = {
            "split": split_id,
            "sample_id": session_id,
            "num_shapes": num_shapes,
            "shape_types": [s["type"] for s in shapes_info],
            "shapes_detail": [],
            "rho": float(rho),
            "min_param": min_param,
            "max_param": max_param,
            "rho_constraint": [
                rho_constraint[0],
                rho_constraint[1] if rho_constraint[1] != float("inf") else "inf",
            ],
            "rejection_attempts": rejection_count,
        }

        # 详细记录每个 shape 的参数与计算的 r_eq
        for i, shape in enumerate(shapes_info):
            shape_type = shape.get("type")
            param = shape.get("param")

            if shape_type == "sphere":
                r_eq = param
                shape_detail = {
                    "id": shape["id"],
                    "type": shape_type,
                    "radius": param,
                    "r_eq": float(r_eq),
                    "center": shape.get("center"),
                }
            elif shape_type == "ellipsoid":
                rx, ry, rz = param
                r_eq = (rx * ry * rz) ** (1 / 3)
                shape_detail = {
                    "id": shape["id"],
                    "type": shape_type,
                    "rx": rx,
                    "ry": ry,
                    "rz": rz,
                    "r_eq": float(r_eq),
                    "center": shape.get("center"),
                }
            elif shape_type == "cube":
                r_eq = param / (2 ** (1 / 3))
                shape_detail = {
                    "id": shape["id"],
                    "type": shape_type,
                    "size": param,
                    "r_eq": float(r_eq),
                    "center": shape.get("center"),
                }
            elif shape_type == "cylinder":
                radius, height = param
                r_eq = (radius**2 * height) ** (1 / 3)
                shape_detail = {
                    "id": shape["id"],
                    "type": shape_type,
                    "radius": radius,
                    "height": height,
                    "r_eq": float(r_eq),
                    "center": shape.get("center"),
                }
            else:
                shape_detail = {"id": shape["id"], "type": shape_type, "param": param}

            metadata["shapes_detail"].append(shape_detail)

        metadata_list.append(metadata)

        # 保存 metadata（JSON 格式）
        metadata_path = os.path.join(config_subdir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 生成仿真配置（保持原有流程）
        source_arr_t = source_arr.transpose(2, 1, 0)  # 转为 zyx 顺序
        source_pattern_in_vol = np.zeros(volume_shape, dtype=np.float32)
        source_pattern_in_vol[
            src_range_z[0] : src_range_z[1],
            src_range_y[0] : src_range_y[1],
            src_range_x[0] : src_range_x[1],
        ] = np.where(source_arr_t > 0.5, 1, 0)

        config_dict = {
            "Domain": {
                "VolumeFile": volume_file,
                "Dim": volume_shape,
                "OriginType": 1,
                "LengthUnit": config.get("lengthunit", 0.1),
                "Media": media_list,
            },
            "Session": {
                "Photons": int(1e7),
                "RNGSeed": generated_count,
                "ID": format_filename(
                    file_naming["normal"]["config"], session_id
                ).replace(".json", ""),
            },
            "Forward": config.get(
                "forward", {"T0": 0.0e00, "T1": 5.0e-08, "DT": 5.0e-08}
            ),
            "Optode": {
                "Source": {
                    "Pos": [src_range_z[0], src_range_y[0], src_range_x[0]],
                    "Dir": [0, 0, 1, "_NaN_"],
                    "Type": "pattern3d",
                    "Pattern": {
                        "Nx": src_voxel_size[0],
                        "Ny": src_voxel_size[1],
                        "Data": source_filename,
                        "Nz": src_voxel_size[2],
                    },
                    "Param1": (src_voxel_size[2], src_voxel_size[1], src_voxel_size[0]),
                }
            },
        }

        json_config_path = os.path.join(
            config_subdir, format_filename(file_naming["normal"]["config"], session_id)
        )
        with open(json_config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        # 无散射配置
        no_scatter_media = deepcopy(media_list)
        for media_item in no_scatter_media:
            media_item["mus"] = 0
        config_no_scatter = deepcopy(config_dict)
        config_no_scatter["Domain"]["Media"] = no_scatter_media
        config_no_scatter["Session"]["ID"] = format_filename(
            file_naming["noscatter"]["config"], session_id
        ).replace(".json", "")

        json_config_nos_path = os.path.join(
            config_subdir,
            format_filename(file_naming["noscatter"]["config"], session_id),
        )
        with open(json_config_nos_path, "w", encoding="utf-8") as f:
            json.dump(config_no_scatter, f, indent=2, ensure_ascii=False)

        generated_count += 1
        sample_id += 1

        if generated_count % 10 == 0:
            print(
                f"  进度: {generated_count}/{num_samples} (rejected: {rejected_count})"
            )

    # 保存 split 汇总 metadata
    split_summary_path = os.path.join(split_dir, f"{split_id}_metadata.json")
    split_summary = {
        "split_id": split_id,
        "split_description": split_cfg.get("description", ""),
        "config": {
            "num_shapes": num_shapes_list,
            "min_param": min_param,
            "max_param": max_param,
            "rho_constraint": [
                rho_constraint[0],
                rho_constraint[1] if rho_constraint[1] != float("inf") else "inf",
            ],
        },
        "total_generated": generated_count,
        "total_rejected": rejected_count,
        "samples": metadata_list,
    }
    with open(split_summary_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ {split_id} 配置生成完成!")
    print(f"  总样本数: {generated_count}")
    print(f"  总拒绝数: {rejected_count}")
    print(f"  输出目录: {split_dir}")
    print(f"  汇总文件: {split_summary_path}")

    # ===== 如果指定了 run_simulation=True，则运行完整的 MCX 仿真和后处理流程 =====
    if run_simulation and generated_count > 0:
        print(
            f"\n【运行仿真】正在对 {split_id} 中的 {generated_count} 个样本进行 MCX 仿真和后处理..."
        )
        try:
            # 运行仿真：使用 split_dir 作为仿真目录
            batch_run_mcx_simulations(split_dir, config=config)
            print(f"  ✓ MCX 仿真完成")

            # 运行后处理
            process_folders(split_dir, config=config)
            print(f"  ✓ 后处理完成")

            print(f"\n✓ {split_id} 仿真与后处理全部完成!")
        except Exception as e:
            print(f"\n✗ {split_id} 仿真或后处理失败: {e}")
            import traceback

            traceback.print_exc()

    return generated_count, rejected_count
