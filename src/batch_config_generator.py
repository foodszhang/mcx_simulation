# --------------------------------------------------
# 体积光学仿真配置生成脚本
# --------------------------------------------------
# 作者：foodszhang@gmail.com
# 说明：统一变量命名风格、增强可读性和文档性，代码功能保持不变
# --------------------------------------------------
import json
import os
from datetime import datetime
import random
import shutil
from copy import deepcopy
from .load_config import load_config
import yaml
from src.shape_generator import generate_multiple_shapes
import numpy as np
from datetime import datetime

# 随机种子由配置文件读取，确保仿真结果可复现
# 注意：config应始终由主流程入口传入，此处禁止文件顶部直接读取！
RANDOM_SEED = 23  # 默认占位，实际运行请传递config
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DATE_STRING = datetime.now().strftime("%Y%m%d")


def generate_multi_blt_config(
    num_configs=200, output_dir=f"./{DATE_STRING}", config=None
):
    """
    批量生成单次光学仿真配置文件，所有参数统一由主流程传入的config字典驱动，避免独立读取。
    :param num_configs: 生成配置的数量
    :param output_dir: 配置文件保存目录
    :param config: 全局配置参数，由main.py主流程传入
    """
    assert config is not None, "必须由主流程指定配置config，禁止在此处独立加载！"
    """
    批量生成单次光学仿真配置文件，所有参数统一由config.yaml读取（见src/load_config.py），方便医学/科研人员调整
    :param num_configs: 生成配置的数量
    :param output_dir: 配置文件保存目录
    :param config: 全局配置参数，默认从文件顶部读取，避免重复读文件，提升效率
    """
    os.makedirs(output_dir, exist_ok=True)
    from .randomize_media import randomize_media

    # --------------------------
    # 从 config.yaml 读取 volume 文件与 shape，材料路径及 media 列表
    # --------------------------
    with open(config["generated_material_yaml_path"], "r", encoding="utf-8") as yf:
        media_list = yaml.safe_load(yf)
    # 拷贝体素文件到 output_dir 根目录，确保配置引用的体素文件在仿真目录内
    volume_shape = config["volume_shape"]
    volume_src_path = config["generated_bin_path"]
    volume_dst_path = os.path.join(output_dir, os.path.basename(volume_src_path))
    if not os.path.exists(volume_src_path):
        raise FileNotFoundError(f"体素文件不存在: {volume_src_path}")
    shutil.copy(volume_src_path, volume_dst_path)

    # 记录相对路径（每个子目录到output_dir根目录），配置文件统一使用 "../xxx.bin"
    volume_file = f"../{os.path.basename(volume_dst_path)}"

    src_range_z = config["src_range"]["z"]
    src_range_y = config["src_range"]["y"]
    src_range_x = config["src_range"]["x"]
    max_rotation = config["src_range"].get("max_rotation", 30)

    def format_filename(template, _id):
        return template.format(id=_id)

    file_naming = config.get("file_naming", {
        "normal": {"config": "{id}.json", "result": "{id}.jnii"},
        "noscatter": {"config": "no_{id}.json", "result": "no_{id}.jnii"}
    })

    for config_idx in range(num_configs):
        session_id = str(config_idx)
        config_subdir = os.path.join(output_dir, session_id)
        os.makedirs(config_subdir, exist_ok=True)
        config_dict = {}

        # --- Domain 设置 ---
        domain = {
            "VolumeFile": volume_file,
            "Dim": volume_shape,
            "OriginType": 1,
            "LengthUnit": config.get("lengthunit", 0.1),
            "Media": media_list,
        }

        # --- Session 参数 ---
        session = {"Photons": int(1e6), "RNGSeed": config_idx, "ID": format_filename(file_naming["normal"]["config"], session_id).replace('.json', '')}
        # --- Forward 参数：由config统一读取 ---
        forward = config.get("forward", {"T0": 0.0e00, "T1": 5.0e-09, "DT": 5.0e-09})

        # --- 光源（Source）参数设定 ---
        source_filename = f"source-{config_idx}.bin"
        src_voxel_size = (
            src_range_x[1] - src_range_x[0],
            src_range_y[1] - src_range_y[0],
            src_range_z[1] - src_range_z[0],
        )
        # 生成三维模式的光源体素数组及形状标签
        source_arr, _ = generate_multiple_shapes(
            src_voxel_size, 1, max_rotation=max_rotation
        )
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
        config_dict["Domain"] = domain
        config_dict["Session"] = session
        config_dict["Forward"] = forward
        config_dict["Optode"] = optode

        # 标准仿真config和结果
        json_config_path = os.path.join(config_subdir, format_filename(file_naming["normal"]["config"], session_id))
        # --- 保存标准仿真配置 ---
        with open(json_config_path, "w") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        # --- 保存无散射配置（mus=0） ---
        no_scatter_media = deepcopy(media_list)
        for media_item in no_scatter_media:
            media_item["mus"] = 0.0
        config_no_scatter = deepcopy(config_dict)
        config_no_scatter["Domain"]["Media"] = no_scatter_media
        # 用noscatter中的模板名，Session.ID也和json名模板一致，不带后缀
        config_no_scatter["Session"]["ID"] = format_filename(file_naming["noscatter"]["config"], session_id).replace('.json', '')
        json_config_nos_path = os.path.join(config_subdir, format_filename(file_naming["noscatter"]["config"], session_id))
        with open(json_config_nos_path, "w") as f:
            json.dump(config_no_scatter, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # 演示：生成 1 个仿真配置
    generate_multi_blt_config(4)
