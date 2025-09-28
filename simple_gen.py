import json
import numpy as np
import nibabel as nib
import os
from scipy.ndimage import rotate


def rotate_x(theta):
    """绕X轴旋转的旋转矩阵"""
    theta_rad = np.radians(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    return np.array([[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]])


def rotate_y(theta):
    """绕Y轴旋转的旋转矩阵"""
    theta_rad = np.radians(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    return np.array([[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]])


def rotate_z(theta):
    """绕Z轴旋转的旋转矩阵"""
    theta_rad = np.radians(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    return np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])


def rotate_point(point, rx=0, ry=0, rz=0):
    """
    对单个点进行三维旋转
    point: 三维坐标 (x, y, z)
    rx, ry, rz: 分别绕X, Y, Z轴的旋转角度(度)
    """
    if rx == 0 and ry == 0 and rz == 0:
        return np.array(point, dtype=np.float64)

    # 转换为numpy数组便于计算
    p = np.array(point, dtype=np.float64)

    # 应用旋转 - 统一旋转顺序为X→Y→Z，与rotate_shape保持一致
    if rx != 0:
        p = np.dot(rotate_x(rx), p)
    if ry != 0:
        p = np.dot(rotate_y(ry), p)
    if rz != 0:
        p = np.dot(rotate_z(rz), p)

    return p


def rotate_shape(shape_array, rx=0, ry=0, rz=0, keep_size=False):
    """
    对三维形状数组进行旋转
    shape_array: 原始三维数组
    rx, ry, rz: 分别绕X, Y, Z轴的旋转角度(度)
    keep_size: 是否保持原始大小，False则自动调整大小以贴合旋转后的物体
    """
    # 如果没有旋转角度，直接返回原数组副本（避免引用问题）
    if rx == 0 and ry == 0 and rz == 0:
        return shape_array.copy(), shape_array.shape

    # 确保输入是正确的三维数组
    if len(shape_array.shape) != 3:
        raise ValueError("输入必须是三维数组")

    # 使用scipy的rotate函数进行旋转，order=1表示线性插值
    rotated = shape_array.copy()

    # 应用旋转 - 统一旋转顺序为X→Y→Z，与rotate_point保持一致
    if rx != 0:
        rotated = rotate(
            rotated,
            angle=rx,
            axes=(1, 2),  # 绕X轴旋转是绕(1,2)轴
            reshape=not keep_size,
            order=1,
            mode="constant",
            cval=0,
        )

    if ry != 0:
        rotated = rotate(
            rotated,
            angle=ry,
            axes=(0, 2),  # 绕Y轴旋转是绕(0,2)轴
            reshape=not keep_size,
            order=1,
            mode="constant",
            cval=0,
        )

    if rz != 0:
        rotated = rotate(
            rotated,
            angle=rz,
            axes=(0, 1),  # 绕Z轴旋转是绕(0,1)轴
            reshape=not keep_size,
            order=1,
            mode="constant",
            cval=0,
        )

    # 将旋转后的值二值化（旋转可能产生中间值）
    rotated = (rotated > 0.5).astype(int)

    # 如果需要，移除全零的边缘以最小化数组大小
    if not keep_size:
        # 找到非零元素的范围
        non_zero = np.where(rotated != 0)

        # 处理完全为空的情况
        if len(non_zero[0]) == 0:
            return np.array([[[0]]]), (1, 1, 1)

        min_x, max_x = np.min(non_zero[0]), np.max(non_zero[0])
        min_y, max_y = np.min(non_zero[1]), np.max(non_zero[1])
        min_z, max_z = np.min(non_zero[2]), np.max(non_zero[2])

        # 裁剪到最小范围
        rotated = rotated[min_x : max_x + 1, min_y : max_y + 1, min_z : max_z + 1]

    return rotated, rotated.shape


def gen_shape(shape, param, rotate_angles):
    source_array = None
    if shape == "sphere":
        radius = param
        size = 2 * radius + 1
        center = radius
        sphere = np.zeros((size, size, size), dtype=int)

        for x in range(size):
            for y in range(size):
                for z in range(size):
                    dx = x - center
                    dy = y - center
                    dz = z - center
                    distance = np.sqrt(dx**2 + dy**2 + dz**2)
                    if distance <= radius:
                        sphere[x, y, z] = 1
        source_array = sphere
    elif shape == "cube":
        size = param
        cube = np.ones((size, size, size), dtype=int)
        source_array = cube
    elif shape == "cylinder":
        # radius, height = param
        radius = 8
        height = 20
        diam = 2 * radius + 1
        center = radius
        cylinder = np.zeros((diam, diam, height), dtype=int)

        for x in range(diam):
            for y in range(diam):
                dx = x - center
                dy = y - center
                distance = np.sqrt(dx**2 + dy**2)
                if distance <= radius:
                    cylinder[x, y, :] = 1

        source_array = cylinder
    source_array, shape = rotate_shape(
        source_array, rotate_angles[0], rotate_angles[1], rotate_angles[2]
    )

    return source_array, shape


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
    # ct_img = nib.load("./ct_data/ct_380x992x208.hdr")
    # ct_data = ct_img.get_fdata().astype(np.uint8)
    # ct_data = np.ascontiguousarray(ct_data)

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
        # 只截取头部
        tag_data = tag_data[100:280, :300, :]
        # ct_data = ct_data[100:280, :300, :]
        # ct_data.tofile("ct_brain.bin")
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
    full_filename = os.path.join(save_dir, filename)
    # simplified_tags = np.transpose(simplified_tags, (2, 1, 0))
    # mcx要求F风格数据
    shapes = list(simplified_tags.shape)
    # f_file = np.ravel(simplified_tags)
    f_file = simplified_tags
    f_file.tofile(full_filename)

    return filename, shapes, media, simplified_tags
