# --------------------------------------------------
# 体积光学仿真配置生成脚本（优化版）
# --------------------------------------------------
# 作者：资深软件与算法工程师
# 说明：统一变量命名风格、增强可读性和文档性，代码功能保持不变
# --------------------------------------------------
import json
import os
from datetime import datetime
import random
import numpy as np
from copy import deepcopy

# 业务相关模块引入
from simple_gen import gen_shape, gen_volume_and_media, generate_multiple_shapes
from vis_3d import visualize_3d_array

# 设置全局随机种子和日期字符串，确保实验可复现
RANDOM_SEED = 23
random.seed(RANDOM_SEED)
DATE_STRING = datetime.now().strftime("%Y%m%d")


def generate_multi_blt_config(num_configs=200, output_dir=f"./{DATE_STRING}"):
    """
    批量生成单次光学仿真配置文件
    :param num_configs: 生成配置的数量
    :param output_dir: 配置文件保存目录
    """
    os.makedirs(output_dir, exist_ok=True)
    from randomize_media import randomize_media

    # 生成体积数据和初始介质参数
    volume_file, volume_shape, media_list, volume_data = gen_volume_and_media(
        "brain", output_dir
    )
    # 随机扰动介质参数（mua、mus），g、n参数保持不变
    perturb_bounds = {"mua": 0.003, "mus": 0.1}
    media_list = randomize_media(media_list, perturb_bounds)

    for config_idx in range(num_configs):
        session_id = str(config_idx)
        config_subdir = os.path.join(output_dir, f"{config_idx}")
        os.makedirs(config_subdir, exist_ok=True)
        config = {}

        # --- Domain 设置 ---
        domain = {
            "VolumeFile": f"../{volume_file}",  # 体素文件相对路径
            "Dim": volume_shape,  # 三维体素尺寸
            "OriginType": 1,  # 原点类型标志
            "LengthUnit": 0.1,  # 单体素实际长度（mm）
            "Media": media_list,  # 光学介质参数列表
        }

        # --- Session 参数 ---
        session = {"Photons": int(1e6), "RNGSeed": config_idx, "ID": session_id}
        # --- Forward 参数 ---
        forward = {"T0": 0.0e00, "T1": 5.0e-09, "DT": 5.0e-09}

        # --- 光源（Source）参数设定 ---
        source_filename = f"source-{config_idx}.bin"
        src_range_z = (60, 120)
        src_range_y = (40, 140)
        src_range_x = (96, 130)
        src_voxel_size = (
            src_range_x[1] - src_range_x[0],
            src_range_y[1] - src_range_y[0],
            src_range_z[1] - src_range_z[0],
        )
        # 生成三维模式的光源体素数组及形状标签
        source_arr, _ = generate_multiple_shapes(src_voxel_size, 1, max_rotation=30)
        full_source_path = os.path.join(config_subdir, source_filename)
        source_arr = source_arr.astype(np.float32)
        source_arr.tofile(full_source_path)
        # 转换为zyx顺序（物理仿真需求）
        source_arr = source_arr.transpose(2, 1, 0)
        # 将光源嵌入到体积数据指定区域，便于后续标签合成
        source_pattern_in_vol = np.zeros(volume_shape, dtype=np.float32)
        source_pattern_in_vol[
            src_range_z[0] : src_range_z[1],
            src_range_y[0] : src_range_y[1],
            src_range_x[0] : src_range_x[1],
        ] = np.where(source_arr > 0.5, 1, 0)
        # --- Optode 光源配置结构 ---
        optode = {
            "Source": {
                "Pos": [src_range_z[0], src_range_y[0], src_range_x[0]],
                # 最后的_NaN_代表光源是各向同性
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
        }

        # --- 组装最终配置字典 ---
        config["Domain"] = domain
        config["Session"] = session
        config["Forward"] = forward
        config["Optode"] = optode
        json_config_path = os.path.join(config_subdir, f"{config_idx}.json")
        json_config_nos_path = os.path.join(config_subdir, f"no_{config_idx}.json")
        # --- 保存标准仿真配置 ---
        with open(json_config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        # --- 保存无散射配置（mus=0） ---
        no_scatter_media = deepcopy(media_list)
        for media_item in no_scatter_media:
            media_item["mus"] = 0.0
        config_no_scatter = deepcopy(config)
        config_no_scatter["Domain"]["Media"] = no_scatter_media
        config_no_scatter["Session"]["ID"] = f"no_{session_id}"
        with open(json_config_nos_path, "w") as f:
            json.dump(config_no_scatter, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # 演示：生成 1 个仿真配置
    generate_multi_blt_config(4)
