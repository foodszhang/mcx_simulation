# -*- coding: utf-8 -*-
"""
photon_shape_inserter.py
-----------------------
医学/生物三维体素数组主流程——内嵌多类光源形状自动化工具模块

主要功能说明：
    - 针对医学三维体素数据，支持自动生成并插入球形、立方体、圆柱体等各类“光源”结构，满足仿真与数据增强需求
    - 支持任意三维空间平移、旋转，保障形状参数灵活与空间不重叠
    - 专为医学成像、光学仿真、深度学习做标准化设计，所有代码与文档均为专业中文实现，便于全团队二次开发与科研落地
    - 单一模块抽象不与主流程耦合，方便多项目/多任务调用
"""

import numpy as np
import random
from scipy.ndimage import rotate


# ------------------- 旋转相关 ------------------
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
    p = np.array(point, dtype=np.float64)
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
    rx, ry, rz: 绕X/Y/Z轴的旋转角度(度)
    keep_size: 是否保持原始体素包围盒尺寸
    """
    if rx == 0 and ry == 0 and rz == 0:
        return shape_array.copy(), shape_array.shape
    if len(shape_array.shape) != 3:
        raise ValueError("输入必须是三维数组")
    rotated = shape_array.copy()
    if rx != 0:
        rotated = rotate(
            rotated,
            angle=rx,
            axes=(1, 2),
            reshape=not keep_size,
            order=1,
            mode="constant",
            cval=0,
        )
    if ry != 0:
        rotated = rotate(
            rotated,
            angle=ry,
            axes=(0, 2),
            reshape=not keep_size,
            order=1,
            mode="constant",
            cval=0,
        )
    if rz != 0:
        rotated = rotate(
            rotated,
            angle=rz,
            axes=(0, 1),
            reshape=not keep_size,
            order=1,
            mode="constant",
            cval=0,
        )
    rotated = (rotated > 0).astype(int)
    if not keep_size:
        non_zero = np.where(rotated != 0)
        if len(non_zero[0]) == 0:
            return np.array([[[0]]]), (1, 1, 1)
        min_x, max_x = np.min(non_zero[0]), np.max(non_zero[0])
        min_y, max_y = np.min(non_zero[1]), np.max(non_zero[1])
        min_z, max_z = np.min(non_zero[2]), np.max(non_zero[2])
        rotated = rotated[min_x : max_x + 1, min_y : max_y + 1, min_z : max_z + 1]
    return rotated, rotated.shape


# ------------------- 体积形状生成 -------------------
def gen_shape(shape, param, rotate_angles):
    """
    生成单个形状（三维球、立方体或圆柱），支持指定参数与旋转
    shape: 形状类型，支持'sphere'/'cube'/'cylinder'
    param: 尺寸参数（球/立方体为半径/边长，圆柱为(半径,高)）
    rotate_angles: 旋转角度元组(x,y,z)
    返回体素数组与其形状三元组
    """
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
        radius, height = param
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
    return source_array, source_array.shape


def generate_multiple_shapes(
    voxel_size,
    num_shapes,
    shape_types=None,
    min_param=4,
    max_param=20,
    max_rotation=360,
):
    """
    在指定三维体素空间内生成多个随机不重叠三维几何体
    支持球、立方体、圆柱体批量生成，所有参数和流程均适配医学/深度学习数据工程标准
    返回三维数组(全体素体)及各形状参数字典列表
    """
    voxel_volume = np.zeros(voxel_size, dtype=int)
    shapes_info = []
    if shape_types is None:
        shape_types = ["sphere", "cube", "cylinder"]
    for shape_id in range(1, num_shapes + 1):
        placed = False
        for _ in range(100):
            shape = random.choice(shape_types)
            # shape = "cylinder"  # 如有特殊需求请在外部参数控制
            if shape == "sphere":
                radius = random.randint(min_param, max_param)
                param = radius
            elif shape == "cube":
                size = random.randint(min_param, max_param)
                param = size
            elif shape == "cylinder":
                radius = random.randint(min_param, max_param)
                height = random.randint(min_param * 2, max_param * 2)
                height = 5 * radius  # 可按需调整参数范围
                param = (radius, height)
            rotate_angles = (
                random.uniform(0, max_rotation),
                random.uniform(0, max_rotation),
                random.uniform(0, max_rotation),
            )
            shape_array, shape_dims = gen_shape(shape, param, rotate_angles)
            max_pos = [voxel_size[i] - shape_dims[i] for i in range(3)]
            if any(dim <= 0 for dim in max_pos):
                continue
            pos = (
                random.randint(0, max_pos[0]),
                random.randint(0, max_pos[1]),
                random.randint(0, max_pos[2]),
            )
            x_slice = slice(pos[0], pos[0] + shape_dims[0])
            y_slice = slice(pos[1], pos[1] + shape_dims[1])
            z_slice = slice(pos[2], pos[2] + shape_dims[2])
            if np.any(voxel_volume[x_slice, y_slice, z_slice] != 0):
                continue
            voxel_volume[x_slice, y_slice, z_slice] = shape_array
            shapes_info.append(
                {
                    "id": shape_id,
                    "type": shape,
                    "param": param,
                    "position": pos,
                    "rotation": rotate_angles,
                    "dimensions": shape_dims,
                }
            )
            placed = True
            break
        if not placed:
            print(f"警告：无法放置第{shape_id}个形状（可能空间不足或尝试次数过多）")
    return voxel_volume, shapes_info
