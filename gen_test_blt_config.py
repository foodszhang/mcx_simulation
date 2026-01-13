import json
import os


from simple_gen import gen_shape, gen_volume_and_media
from vis_3d import visualize_3d_array
from simple_gen import generate_multiple_shapes


from datetime import datetime
import random
import numpy as np


def _infer_target_label_from_volfile(volfile: str) -> int:
    return 1


def _try_place_source_within_mask(
    source_zyx: np.ndarray, allowed_mask_zyx: np.ndarray, tries: int = 200
) -> np.ndarray | None:
    coords = np.argwhere(source_zyx > 0)
    if coords.size == 0:
        return np.zeros_like(allowed_mask_zyx, dtype=source_zyx.dtype)

    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    crop = source_zyx[z0:z1, y0:y1, x0:x1]

    dz, dy, dx = crop.shape
    mz, my, mx = allowed_mask_zyx.shape
    if dz > mz or dy > my or dx > mx:
        return None

    for _ in range(tries):
        oz = random.randint(0, mz - dz)
        oy = random.randint(0, my - dy)
        ox = random.randint(0, mx - dx)
        sub = allowed_mask_zyx[oz : oz + dz, oy : oy + dy, ox : ox + dx]
        if np.all(sub[crop > 0]):
            out = np.zeros_like(allowed_mask_zyx, dtype=source_zyx.dtype)
            out[oz : oz + dz, oy : oy + dy, ox : ox + dx] = crop
            return out

    return None


# 获取当前日期
current_date = datetime.now()

# 格式化为 ymd 形式（例如：20250924）
today_ymd = current_date.strftime("%Y%m%d")
random.seed(42)


def gen_multi_single_blt_config(num=200, save_dir=f"./{today_ymd}"):
    os.makedirs(save_dir, exist_ok=True)
    volfile, vol_shape, media, vol = gen_volume_and_media("abdomen", save_dir)
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
        # range_z = (50, 120)
        # range_y = (160, 280)
        # range_x = (96, 140)
        range_z = (30, 280)
        range_y = (170, 250)
        range_x = (15, 280)

        rz0, rz1 = max(0, range_z[0]), min(range_z[1], vol_shape[0])
        ry0, ry1 = max(0, range_y[0]), min(range_y[1], vol_shape[1])
        rx0, rx1 = max(0, range_x[0]), min(range_x[1], vol_shape[2])

        voxel_size = (rx1 - rx0, ry1 - ry0, rz1 - rz0)
        full_source_filename = os.path.join(each_save_dir, source_filename)

        target_label = _infer_target_label_from_volfile(volfile)
        allowed_mask = vol[rz0:rz1, ry0:ry1, rx0:rx1] == target_label
        if not np.any(allowed_mask):
            raise Exception(
                f"ROI内找不到label={target_label} (volfile={volfile}), 请调整range或label映射"
            )

        # Debug: visualize ROI and allowed region (labels here are for inspection only)
        roi_label = np.uint8(250)
        allowed_label = np.uint8(251)
        roi_tag = vol.astype(np.uint8).copy()
        roi_tag[rz0:rz1, ry0:ry1, rx0:rx1] = roi_label
        roi_tag.tofile(os.path.join(each_save_dir, "roi_tag.bin"))
        print("!!!!!!", roi_tag.shape)

        roi_allowed_tag = vol.astype(np.uint8).copy()
        roi_allowed_tag[rz0:rz1, ry0:ry1, rx0:rx1] = np.where(
            allowed_mask, allowed_label, roi_label
        ).astype(np.uint8)
        roi_allowed_tag.tofile(os.path.join(each_save_dir, "roi_allowed_tag.bin"))

        placed_zyx = None
        for _ in range(50):
            source_xyz, shapes = generate_multiple_shapes(
                voxel_size, 4, max_rotation=30
            )
            source_zyx = source_xyz.astype(np.float32).transpose(2, 1, 0)
            placed_zyx = _try_place_source_within_mask(source_zyx, allowed_mask)
            if placed_zyx is not None:
                break
        if placed_zyx is None:
            raise Exception("无法在ROI的指定label区域内放置形状(尝试次数耗尽)")

        placed_zyx.transpose(2, 1, 0).astype(np.float32).tofile(full_source_filename)

        ###TODO: 更智能的选择
        # 区域

        source_in_vol = np.zeros(vol_shape, dtype=np.float32)
        source_in_vol[rz0:rz1, ry0:ry1, rx0:rx1] = placed_zyx
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
                "Pos": [rx0, ry0, rz0],
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
                "Param1": (voxel_size[2], voxel_size[1], voxel_size[0]),
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
