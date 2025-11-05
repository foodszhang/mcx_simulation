# -*- coding: utf-8 -*-
"""
医学/生物仿真实验-体素与材料参数、配置主流程（专为rat小鼠/脑区自适应生成）
--------------------------------------------------------
- 专责体素（bin）、材料参数YAML整体生成，主流程逻辑入口。
- 专业中文注释，全部参数由config.yaml自动管理。
- 不含底层三维形状/旋转等实现，全部委托shape_generator等公用模块，保障主流程纯净。
- 调用方式：uv run python src/tool/gen_voxel_and_config.py
作者: foodszhang@gmail.com
"""

import os
import yaml
import numpy as np
import nibabel as nib
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.load_config import load_config


def gen_volume_and_media(area, bin_dir="./", yaml_dir="./"):
    """
    按区域生成体素bin及材料yaml，输出到bin_dir/yaml_dir
    只保留IO和主流程，shape部分已抽象到shape_generator.py
    """
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)
    if not os.path.exists(yaml_dir):
        os.makedirs(yaml_dir)

    # 从原始解剖数据加载，判定区域
    img = nib.load("./ct_data/atlas_380x992x208.hdr")
    tag_data = img.get_fdata().astype(np.uint8)
    tag_data = np.ascontiguousarray(tag_data)
    # 若为4维，去除最后一维，只取第一通道（常见于Nifti/Analyze格式）
    if len(tag_data.shape) == 4:
        tag_data = tag_data[:, :, :, 0]

    if area == "brain":
        filename = "volume_brain.bin"
        tag_data = tag_data[100:280, 116:278, :]
        tag_data = np.pad(
            tag_data,
            pad_width=((1, 1), (1, 1), (1, 1)),
            mode="constant",
            constant_values=0,
        )
        tag_mapping = {0: 0, 1: 1, 2: 2, 3: 1, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 10: 3}
        media = [
            {"mua": 0.00, "mus": 0.0, "g": 1.00, "n": 1.0},
            {"mua": 0.0921, "mus": 0.619, "g": 0.9, "n": 1.37},
            {"mua": 0.02083, "mus": 2.490, "g": 0.9, "n": 1.37},
            {"mua": 0.0648, "mus": 2.392, "g": 0.9, "n": 1.37},
        ]
    else:
        raise Exception("当前仅支持area=brain，如需其他区域请补充实现")

    simplified_tags = np.zeros_like(tag_data)
    for original, simplified in tag_mapping.items():
        simplified_tags[tag_data == original] = simplified
    unique_tags = np.unique(tag_data)
    for tag in unique_tags:
        if tag not in tag_mapping:
            print(f"警告: 发现未映射的标签 {tag}，已默认设为皮肤肌肉(1)")
            simplified_tags[tag_data == tag] = 1

    full_bin_filename = os.path.join(bin_dir, filename)
    shapes = list(simplified_tags.shape)
    simplified_tags.tofile(full_bin_filename)

    # ==========生成材料参数YAML==========
    material_yaml_header = (
        "# 小鼠（rat）" + area + "区域医学仿真的近红外二区（NIR-II）标准材料参数\n"
        "# 不同编号代表不同组织类型，具体区域参考README及项目说明\n"
        "# 单位：mua, mus [1/mm]，折射率n [-]，各项性因子g [-]，参数详见工程文档说明\n"
        "# 本文件由主流程自动生成，严禁手动修改\n"
    )
    config = load_config()
    species = config.get("species", "rat")
    wavelength = config.get("wavelength", 780)
    material_yaml_filename = f"{species}_{area}_{wavelength}nm.yaml"
    material_yaml_path = os.path.join(yaml_dir, material_yaml_filename)
    material_yaml_content = yaml.dump(
        media, allow_unicode=True, sort_keys=False, default_flow_style=False
    )
    with open(material_yaml_path, "w", encoding="utf-8") as matf:
        matf.write(material_yaml_header)
        matf.write(material_yaml_content)
    return filename, shapes, media, simplified_tags


if __name__ == "__main__":
    """
    主流程入口：
    从config.yaml加载区域/输出目录等参数，自动分目录生成bin及yaml
    输入/输出流程全自动日志输出，方便科研建模和复现。
    """
    config = load_config()
    area = config.get("area", "brain")
    base_input_dir = config.get("base_input_dir", "./volume_bases")
    bin_dir = os.path.join(base_input_dir, "bin")
    yaml_dir = os.path.join(base_input_dir, "material")
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)
    if not os.path.exists(yaml_dir):
        os.makedirs(yaml_dir)
    print(
        f"[仿真主流程启动] 区域: {area}, BIN输出: {bin_dir}, 材料YAML输出: {yaml_dir} (volume_bases基础输入目录)"
    )
    filename, shapes, media, simplified_tags = gen_volume_and_media(
        area, bin_dir, yaml_dir
    )
    bin_path = os.path.abspath(os.path.join(bin_dir, filename))
    material_yaml_filename = (
        f"{config.get('species', 'rat')}_{area}_{config.get('wavelength',780)}nm.yaml"
    )
    yaml_path = os.path.abspath(os.path.join(yaml_dir, material_yaml_filename))
    print(f"[生成成功] 体素文件: {bin_path}, 尺寸: {shapes}, 材料文件: {yaml_path}")

    # =========写入config.yaml供后续流程自动读取=========
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    )
    with open(config_path, "r", encoding="utf-8") as cf:
        all_config = yaml.safe_load(cf)
    all_config["generated_bin_path"] = bin_path
    all_config["generated_material_yaml_path"] = yaml_path
    with open(config_path, "w", encoding="utf-8") as cf:
        yaml.dump(all_config, cf, allow_unicode=True, sort_keys=False)
    print(f"[配置同步] 已将bin和材料参数yaml路径写入config.yaml，供下游全流程读取！")
# 注意：主入口不允许return，流程仅日志输出，符合科研工程开发规范。
