import json
import os


from simple_gen import gen_shape, gen_volume_and_media


from datetime import datetime
import random
import numpy as np

# 获取当前日期
current_date = datetime.now()

# 格式化为 ymd 形式（例如：20250924）
today_ymd = current_date.strftime("%Y%m%d")
random.seed(42)


def gen_multi_single_blt_config(num=200, save_dir=f"./{today_ymd}"):
    os.makedirs(save_dir, exist_ok=True)
    volfile, vol_shape, media = gen_volume_and_media("brain", save_dir)
    for i in range(num):
        session = str(i)
        each_save_dir = os.path.join(save_dir, f"{i}")
        os.makedirs(each_save_dir, exist_ok=True)
        # random = random.randint(num)
        config = {}

        Domain = {
            # 二进制体素文件, 体素值与media对应，或者json定义的shapes文件
            # "VolumeFile": 'volume.bin',
            # TODO: 这里显式指定为上级目录下
            "VolumeFile": f"../{volfile}",
            "Dim": vol_shape,
            "OriginType": 1,
            # 一个体素对应实际距离， 单位(mm)
            "LengthUnit": 0.1,
            "Media": media,
        }
        Session = {
            # 光子数
            "Photons": int(1e6),
            # 随机数种子
            "RNGSeed": i,
            "ID": session,
        }
        Forward = {
            "T0": 0.0e00,
            "T1": 5.0e-09,
            "DT": 5.0e-09,
        }

        radius = random.randint(1, 14)

        source_type = "sphere"
        # print("55555", sphsrc.shape)
        source_filename = f"{source_type}-{radius}.bin"
        full_source_filename = os.path.join(each_save_dir, source_filename)
        source, source_shape = gen_shape(radius, source_type)
        if not os.path.exists(full_source_filename):
            source.tofile(full_source_filename)

        ###TODO: 更智能的选择
        rand_x = random.randint(100, 120)
        rand_y = random.randint(140, 260)
        rand_z = random.randint(100, 120)
        source_in_vol = np.zeros(vol_shape, dtype=np.float32)
        source_in_vol[
            rand_x : rand_x + source_shape[0],
            rand_y : rand_y + source_shape[1],
            rand_z : rand_z + source_shape[2],
        ] = source
        source_in_vol_filename = "source_in_vol.npy"
        # print("55555", source_in_vol.dtype, source_in_vol.shape)
        full_source_in_vol_filename = os.path.join(
            each_save_dir, source_in_vol_filename
        )

        # source
        np.save(full_source_in_vol_filename, source_in_vol)
        Optode = {
            "Source": {
                "Pos": [rand_x, rand_y, rand_z],
                "Dir": [0, 0, 1],
                "Type": "pattern3d",
                # 光源维度
                "Pattern": {
                    "Nx": source_shape[0],
                    "Ny": source_shape[1],
                    "Nz": source_shape[2],
                    "Data": f"{source_filename}",
                },
                # 光源在维度下的分布， 值代表权重
                "Param1": source_shape,
            }
        }
        config["Domain"] = Domain
        config["Session"] = Session
        config["Forward"] = Forward
        config["Optode"] = Optode
        save_file = os.path.join(each_save_dir, f"{i}.json")
        with open(save_file, "w") as f:
            json.dump(config, f)


if __name__ == "__main__":
    gen_multi_single_blt_config(1)
