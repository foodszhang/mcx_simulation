import json
import os


from simple_gen import gen_shape, gen_volume_and_media
from vis_3d import visualize_3d_array
from simple_gen import generate_multiple_shapes


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
    volfile, vol_shape, media, vol = gen_volume_and_media("brain", save_dir)
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

        source_filename = f"source-{i}.bin"
        range_x = (50, 120)
        range_y = (160, 280)
        range_z = (96, 140)
        voxel_size = (
            range_x[1] - range_x[0],
            range_y[1] - range_y[0],
            range_z[1] - range_z[0],
        )
        source, shapes = generate_multiple_shapes(voxel_size, 4, max_rotation=30)
        print("66666", len(shapes))
        full_source_filename = os.path.join(each_save_dir, source_filename)
        source = source.astype(np.float32)
        source.tofile(full_source_filename)
        print("566666", source.dtype, source.shape)

        ###TODO: 更智能的选择
        # 区域

        source_in_vol = np.zeros(vol_shape, dtype=np.float32)
        source_in_vol[
            range_x[0] : range_x[1],
            range_y[0] : range_y[1],
            range_z[0] : range_z[1],
        ] = source
        source_in_vol_filename = "source_in_vol.npy"
        full_source_in_vol_filename = os.path.join(
            each_save_dir, source_in_vol_filename
        )

        # source
        np.save(full_source_in_vol_filename, source_in_vol)

        all_in_one = np.zeros_like(vol)
        all_in_one = np.where(source_in_vol > 0, 4, vol)
        all_in_one = all_in_one.astype(np.uint8)
        all_in_one.tofile(os.path.join(each_save_dir, "all_tag.bin"))
        # visualize_3d_array(all_in_one)

        Optode = {
            "Source": {
                "Pos": [range_x[0], range_y[0], range_z[0]],
                "Dir": [0, 0, 1, "_NaN_"],
                "Type": "pattern3d",
                # 光源维度
                "Pattern": {
                    "Nx": voxel_size[0],
                    "Ny": voxel_size[1],
                    "Nz": voxel_size[2],
                    "Data": f"{source_filename}",
                },
                # 光源在维度下的分布， 值代表权重
                "Param1": voxel_size,
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
