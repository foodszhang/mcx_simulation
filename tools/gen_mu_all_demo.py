"""
    示例脚本：批量根据体素与配置自动生成mu_all矩阵，并利用无散射投影进行衰减校正。
    —— 医学、生物场景专用，全部注释使用中文。

    使用说明：
        - 所有依赖请用 uv add numpy pyyaml
        - 运行脚本请用 uv run python tools/gen_mu_all_demo.py
        - 输入参数均根据config.yaml和指定数据集文件夹自动解析
"""

import os
import numpy as np
import yaml


# 加载配置（yaml格式）
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_material(mat_yaml_path, scatter_mode=False):
    """
    加载rat_brain_1200nm.yaml格式的材质参数，支持无散射吸收系数为mua+mus。输入格式严格为list of dict，每个dict包含mua、mus等参数。
    若scatter_mode=True，则吸收系数按mua+mus加和。
    返回材质吸收系数数组。
    """
    with open(mat_yaml_path, "r", encoding="utf-8") as f:
        mat_list = yaml.safe_load(f)
    mua_list = []
    for mat_item in mat_list:
        if scatter_mode:
            mua_list.append(mat_item["mua"] + mat_item["mus"])
        else:
            mua_list.append(mat_item["mua"])
    return np.array(mua_list)


# ===== 核心DDA体素遍历和mu_all生成 =====
from src.attenuation_correction import (
    attenuation_mu_numpy,
    attenuation_projection_correction_numpy,
)


def main(config_path, dataset_dir):
    # 中文注释：解析配置文件及路径
    config = load_config(config_path)
    volume_path = config["generated_bin_path"]
    mat_yaml_path = config["generated_material_yaml_path"]
    no_depth_proj_key = config["projection"]["no_depth_proj_npz"]
    no_proj_key = config["projection"]["no_flux_proj_npz"]
    voxel_shape = config["volume_shape"]
    voxel_size_mm = float(config.get("lengthunit", 1))
    rotation_deg = (
        config["projection"]["angles"][0] if "angles" in config["projection"] else 0
    )
    z_virtual = 0.0  # 设为0，如需变更请按实际配置
    # 体素分割体和材料吸收系数读取（此处用npy或自定义格式，需根据实际格式拓展）
    # 加载体积数据
    vol = np.memmap(volume_path, dtype=np.uint8, mode="r", shape=tuple(voxel_shape))
    # 加载材料参数，假定mua_list长度等于可能的label数
    mua_list = load_material(mat_yaml_path, scatter_mode=True)
    # 默认吸收系数设为材料1的mua(一般是最外侧的组织)
    mu_default = mua_list[1]
    # 由分割体得到每个体素的mua（假定体素值=材料label）
    mu_3d = mua_list[vol.astype(np.int32)]
    # 获取第0组的深度投影和无散射投影路径
    depth_proj_path = os.path.join(dataset_dir, "0", no_depth_proj_key)
    proj_img_path = os.path.join(dataset_dir, "0", no_proj_key)
    # 加载深度、投影数据，允许npz为多角度字典
    depth_npz = np.load(depth_proj_path, allow_pickle=True)
    proj_npz = np.load(proj_img_path, allow_pickle=True)
    # 遍历所有角度
    for angle_key in depth_npz.files:
        depth_map = depth_npz[angle_key]
        proj_img = proj_npz[angle_key]
        # === 用当前角度的深度图生成mu_all矩阵 ===
        mu_all = attenuation_mu_numpy(
            mu_3d, depth_map, float(angle_key), voxel_size_mm, z_virtual, mu_default
        )
        # 用无散射投影图进行荧光强度衰减校正
        result = attenuation_projection_correction_numpy(proj_img, mu_all)
        # 可视化校正前投影与校正后投影，使用‘hot’色图对比保存
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(proj_img, cmap="hot")
        axs[0].set_title("校正前投影")
        axs[1].imshow(result, cmap="hot")
        axs[1].set_title("校正后投影")
        plt.tight_layout()
        cmp_path = os.path.join(dataset_dir, "0", f"compare_{angle_key}.png")
        plt.savefig(cmp_path)
        plt.close()
        print(f"已保存角度{angle_key}的校正前后对比图：{cmp_path}")


if __name__ == "__main__":
    # 默认参数设定，用户不必每次命令行输入
    default_config_path = "config/config.yaml"
    default_dataset_dir = "20251106"
    import sys

    if len(sys.argv) >= 3:
        config_path = sys.argv[1]
        dataset_dir = sys.argv[2]
    else:
        config_path = default_config_path
        dataset_dir = default_dataset_dir
        print(
            f"未检测到命令行参数，自动采用默认配置: {config_path}, 数据目录: {dataset_dir}"
        )
    main(config_path, dataset_dir)
