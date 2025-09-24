import json
import numpy as np
import nibabel as nib
import os


def gen_shape(radius, shape):
    xi, yi, zi = np.meshgrid(
        np.arange(-radius, radius + 1),
        np.arange(-radius, radius + 1),
        np.arange(-radius, radius + 1),
        indexing="ij",
    )
    sphsrc = None
    if shape == "sphere":
        sphsrc = (xi**2 + yi**2 + zi**2) <= radius**2
        sphsrc = sphsrc.astype(np.int32)
        sphsrc = sphsrc.astype(np.float32)
    sphsrc_shape = sphsrc.shape
    # print("!!!!!", sphsrc.flags)
    sphsrc = np.ravel(sphsrc, order="F")

    return sphsrc, sphsrc_shape


def gen_volume_and_media(area, save_dir="./"):
    """
    原始对应
    0 --> 背景
    1 --> 皮肤
    2 --> 骨骼
    3 --> 眼睛
    4 --> 延髓
    5 --> 小脑
    6 --> 嗅球
    7 --> 外部大脑
    8 --> 纹状体
    9 --> 心脏
    10 --> 大脑其他部分
    4+5+6+7+8+10 --> 全脑
    11 --> 咬肌
    12 --> 泪腺
    13 --> 膀胱
    14 --> 睾丸
    15 --> 胃
    16 --> 脾脏
    17 --> 胰腺
    18 --> 肝脏
    19 --> 肾脏
    20 --> 肾上腺
    21 --> 肺
    """
    img = nib.load("./ct_data/atlas_380x992x208.hdr")
    tag_data = img.get_fdata().astype(np.uint8)
    tag_data = np.ascontiguousarray(tag_data)
    # print("5555", tag_data.flags)

    media = []
    if len(tag_data.shape) == 4:
        tag_data = tag_data[:, :, :, 0]
    if area == "full":
        filename = "volume_full.bin"
        tag_mapping = {
            0: 0,  # 背景
            1: 1,  # 皮肤
            2: 2,  # 骨骼
            3: 1,  # 眼睛 -> 1
            4: 3,  # 延髓 -> 3
            5: 3,  # 小脑 -> 3
            6: 3,  # 嗅球 -> 3
            7: 3,  # 外部大脑 -> 3
            8: 3,  # 纹状体 -> 3
            9: 4,  # 心脏 -> 4
            10: 3,  # 大脑其他部分 -> 3
            15: 6,  # 胃 -> 6
            # 16: 7,  # 脾脏 -> 7
            # 17: 8,  # 胰腺 -> 8
            18: 7,  # 肝脏 -> 8
            19: 8,  # 肾脏 -> 8
            21: 9,  # 肺 -> 9
        }
        media = [
            # 0: 背景
            {"mua": 0.00, "mus": 0.0, "g": 1.00, "n": 1.0},
            # 1: 皮肤及相关组织（皮肤、眼睛、咬肌、泪腺、膀胱、睾丸、肾上腺）
            {"mua": 0.0338, "mus": 11.9827, "g": 0.9, "n": 1.37},
            # 2: 骨骼
            {"mua": 0.05251, "mus": 24.4153, "g": 0.9, "n": 1.37},
            # 3: 全脑（延髓、小脑、嗅球、外部大脑、纹状体、大脑其他部分）
            {"mua": 0.03180, "mus": 15.9590, "g": 0.9, "n": 1.37},
            # 4: 心脏
            {"mua": 0.065, "mus": 22.4, "g": 0.9, "n": 1.38},
            # 6: 胃
            {"mua": 0.028, "mus": 18.7, "g": 0.88, "n": 1.37},
            # 7: 脾脏
            {"mua": 0.052, "mus": 25.3, "g": 0.89, "n": 1.38},
            # 8: 胰腺
            {"mua": 0.035, "mus": 16.2, "g": 0.87, "n": 1.37},
            # 9: 肝脏
            {"mua": 0.041, "mus": 20.5, "g": 0.88, "n": 1.37},
            # 10: 肾脏
            {"mua": 0.038, "mus": 19.2, "g": 0.89, "n": 1.37},
            # 11: 肺
            {"mua": 0.022, "mus": 35.7, "g": 0.86, "n": 1.38},
        ]
    elif area == "brain":
        filename = "volume_brain.bin"
        filename_c = "volume_brain.npy"
        # 只截取头部
        tag_data = tag_data[73:301, :300, :]

        tag_mapping = {
            0: 0,  # 背景
            1: 1,  # 皮肤
            2: 2,  # 骨骼
            3: 1,  # 眼睛 -> 1
            4: 3,  # 延髓 -> 3
            5: 3,  # 小脑 -> 3
            6: 3,  # 嗅球 -> 3
            7: 3,  # 外部大脑 -> 3
            8: 3,  # 纹状体 -> 3
            10: 3,  # 大脑其他部分 -> 3
        }
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

    else:
        raise Exception("Unsupport area")
    simplified_tags = np.zeros_like(tag_data)

    # 应用映射关系
    for original, simplified in tag_mapping.items():
        simplified_tags[tag_data == original] = simplified
    # if os.path.exists(filename):
    # data.astype(np.uint8).tofile('volume_full.bin')
    unique_tags = np.unique(tag_data)
    for tag in unique_tags:
        if tag not in tag_mapping:
            print(f"警告: 发现未映射的标签 {tag}，已默认设为皮肤肌肉(1)")
            simplified_tags[tag_data == tag] = 1
    filename = os.path.join(save_dir, filename)
    # simplified_tags = np.transpose(simplified_tags, (2, 1, 0))
    # mcx要求F风格数据
    shapes = list(simplified_tags.shape)
    f_file = np.ravel(simplified_tags, order="F")
    f_file.tofile(filename)
    c_filename = os.path.join(save_dir, filename_c)
    # shapes.reverse()
    c_file = np.ravel(simplified_tags, order="C")
    c_file.tofile(filename_c)

    return filename, shapes, media
