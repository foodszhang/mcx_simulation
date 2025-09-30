import json
import os

import subprocess

import shutil


from datetime import datetime
import random

# 获取当前日期

# 格式化为 ymd 形式（例如：20250924）
random.seed(42)
media = [
    # 0: 背景
    {"mua": 0.00, "mus": 0.0, "g": 1.00, "n": 1.0},
    # 1: 皮肤及相关组织（皮肤、眼睛、咬肌、泪腺、膀胱、睾丸、肾上腺）
    {"mua": 0.0338, "mus": 11.9827, "g": 0.9, "n": 1.37},
    # 2: 骨骼
    {"mua": 0.05251, "mus": 24.4153, "g": 0.9, "n": 1.37},
    # 3: 全脑（延髓、小脑、嗅球、外部大脑、纹状体、大脑其他部分）
    {"mua": 0.03180, "mus": 15.9590, "g": 0.9, "n": 1.37},
]


def gen_green_blt_config(size=50):
    save_dir = f"./green_{size}"
    os.makedirs(save_dir, exist_ok=True)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                session = f"{x}_{y}_{z}"
                each_save_dir = os.path.join(save_dir, session)
                # random = random.randint(num)
                config = {}
                shutil.copy("volume.json", save_dir)

                Domain = {
                    # 二进制体素文件, 体素值与media对应，或者json定义的shapes文件
                    # "VolumeFile": 'volume.bin',
                    # TODO: 这里显式指定为上级目录下
                    "VolumeFile": "volume.json",
                    "Dim": (size, size, size),
                    "OriginType": 1,
                    # 一个体素对应实际距离， 单位(mm)
                    "LengthUnit": 0.1,
                    "Media": media,
                }
                Session = {
                    # 光子数
                    "Photons": int(1e6),
                    # 随机数种子
                    "RNGSeed": 42,
                    "ID": session,
                }
                Forward = {
                    "T0": 0.0e00,
                    "T1": 5.0e-09,
                    "DT": 5.0e-09,
                }
                # source
                Optode = {
                    "Source": {
                        "Pos": [x, y, z],
                        "Dir": [0, 0, 1, "_NaN_"],
                        "Type": "isotropic",
                        # 光源在维度下的分布， 值代表权重
                    }
                }
                config["Domain"] = Domain
                config["Session"] = Session
                config["Forward"] = Forward
                config["Optode"] = Optode
                json_file = f"{x}_{y}_{z}.json"
                save_file = os.path.join(save_dir, json_file)
                with open(save_file, "w") as f:
                    json.dump(config, f)
                result = subprocess.run(
                    ["mcxcl", "-f", json_file, "-a", "1"],
                    cwd=save_dir,  # 在子文件夹中执行命令
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(f"命令成功完成，输出: {result.stdout}")


if __name__ == "__main__":
    gen_green_blt_config(size=50)
