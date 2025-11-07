"""
    本模块实现平行投影下 "荧光强度衰减补偿 + 默认吸收系数扣除" 两套算法：
    1. torch版本（深度学习训练/推断场景，GPU加速）
    2. numpy+numba版本（体素衰减预计算/缓存，批量静态场景高效）
    —— 医学与生物工程专用，所有注释与文档均为中文。

核心相机方向与旋转逻辑：
- 主视角为z轴（相机由物体顶部朝下观察，即z值从大到小遍历）。
- rotation_deg参数指定以y轴穿过体素中心为旋转轴的侧视角度，主视角为0°，正值为顺时针。
- 相机方向矢量通过rotation_matrix_y(rotation_deg)和[0,0,-1]实现（见gen_mul_projection主流程）。
- 该方向矢量控制所有DDA遍历与衰减计算方向，保证与投影一致。

输入参数说明：
- proj_img: 二维荧光投影矩阵
- mu_3d: 三维体素吸收系数矩阵
- depth_map: 每个投影点对应入射深度
- rotation_deg: 投影角度
- voxel_size_mm: 每个体素长度（mm）
- z_virtual: 虚拟Z平面深度
- mu_default: 默认吸收系数

输出：
- 校正后的二维荧光投影矩阵
- 或预先批量计算并缓存好的mu_all衰减矩阵
"""

import torch
import numpy as np
from numba import njit, prange
from src.gen_mul_projection import rotation_matrix_y

# ==== torch版本函数（深度学习专用，支持GPU）====


def attenuation_mu_tensor(
    mu_3d, depth_map, rotation_deg, voxel_size_mm, z_virtual, mu_default
):
    """
    批量计算所有像素的衰减系数mu_all（不依赖proj_img，仅依赖体素结构和几何参数），torch张量并行加速。
    相机方向通过rotation_matrix_y(rotation_deg)和[0,0,-1]生成，确保与gen_mul_projection主流程完全一致。
    可直接缓存结果用于后续多次校正。
    """
    device = mu_3d.device if isinstance(mu_3d, torch.Tensor) else torch.device("cpu")
    mu_3d = torch.as_tensor(mu_3d, device=device).float()
    depth_map = torch.as_tensor(depth_map, device=device).float()
    h, w = depth_map.shape
    nx, ny, nz = mu_3d.shape
    # 计算投影点批量世界坐标
    yv, xv = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    x0 = (xv - h / 2 + 0.5) * voxel_size_mm
    y0 = (yv - w / 2 + 0.5) * voxel_size_mm
    z0 = torch.zeros_like(x0)
    start_points = torch.stack([x0, y0, z0], dim=-1).reshape(-1, 3)
    # 相机方向：与gen_mul_projection主流程保持一致

    direction_np = np.matmul(
        rotation_matrix_y(rotation_deg), np.array([0, 0, -1], dtype=np.float32)
    )
    direction = torch.as_tensor(direction_np, device=device).float()
    # 批量处理每个投影点（可进一步并行化/优化）
    mu_all = torch.zeros(h * w, device=device)
    depth_flat = depth_map.reshape(-1)
    for i in range(h * w):
        curr_pos = start_points[i]
        curr_idx = torch.floor(curr_pos / voxel_size_mm).long().tolist()
        dep = float(depth_flat[i])
        raysum = 0.0
        # DDA遍历（下步可进一步并行/CUDA化）
        dist = 0.0
        pos = curr_pos.clone().cpu().numpy()
        idx = np.floor(pos / voxel_size_mm).astype(int)
        direction_np = direction.cpu().numpy()
        while dist < dep:
            if not (0 <= idx[0] < nx and 0 <= idx[1] < ny and 0 <= idx[2] < nz):
                break
            mu_here = float(mu_3d[idx[0], idx[1], idx[2]])
            next_boundary = (idx + (direction_np > 0)) * voxel_size_mm
            delta = (next_boundary - pos) / direction_np
            delta[direction_np == 0] = np.inf
            step_len = np.min(delta[delta > 0])
            if step_len < 1e-6:
                step_len = dep - dist
            if dist + step_len > dep:
                step_len = dep - dist
            raysum += mu_here * step_len
            pos += direction_np * step_len
            idx = np.floor(pos / voxel_size_mm).astype(int)
            dist += step_len
            if step_len <= 0 or dist >= dep:
                break
        # 虚拟面外补偿
        if dep >= z_virtual:
            raysum += mu_default * (dep - z_virtual)
        else:
            raysum -= mu_default * (z_virtual - dep)
        mu_all[i] = raysum
    mu_all = mu_all.reshape(h, w)
    return mu_all


def attenuation_projection_correction_torch(proj_img, mu_all):
    """
    用预先批量计算的mu_all矩阵对任意proj_img做校正，支持pytorch训练或批量推断。
    """
    device = (
        proj_img.device if isinstance(proj_img, torch.Tensor) else torch.device("cpu")
    )
    proj_img = torch.as_tensor(proj_img, device=device).float()
    mu_all = torch.as_tensor(mu_all, device=device).float()
    return proj_img * torch.exp(mu_all)


# ==== numpy+numba版本函数（预计算缓存专用，高效静态应用）====


@njit(cache=True, parallel=True)
def dda_voxel_tracing_numba(start, direction, grid_shape, max_dist, voxel_size):
    """
    DDA体素遍历算法（单像素版），已用numba加速，仅依赖numpy输入
    相机方向部分与主流程一致（rotation_matrix_y(rotation_deg)和[0,0,-1]）。
    """
    direction = direction / np.sqrt(np.sum(direction**2))
    nx, ny, nz = grid_shape
    curr_pos = np.array(start, dtype=np.float32)
    curr_idx = np.floor(curr_pos / voxel_size).astype(np.int32)
    steps = []
    lengths = []
    dist = 0.0
    while dist < max_dist:
        if not (
            0 <= curr_idx[0] < nx and 0 <= curr_idx[1] < ny and 0 <= curr_idx[2] < nz
        ):
            break
        steps.append((curr_idx[0], curr_idx[1], curr_idx[2]))
        next_boundary = (curr_idx + (direction > 0)) * voxel_size
        delta = (next_boundary - curr_pos) / direction
        for j in range(3):
            if direction[j] == 0:
                delta[j] = np.inf
        step_len = np.min(delta[delta > 0])
        if step_len < 1e-6:
            step_len = max_dist - dist
        if dist + step_len > max_dist:
            step_len = max_dist - dist
        lengths.append(step_len)
        curr_pos += direction * step_len
        curr_idx = np.floor(curr_pos / voxel_size).astype(np.int32)
        dist += step_len
        if step_len <= 0 or dist >= max_dist:
            break
    return steps, lengths


@njit(cache=True, parallel=True)
def attenuation_mu_numpy(
    mu_3d, depth_map, rotation_deg, voxel_size_mm, z_virtual, mu_default
):
    """
    批量计算所有像素的衰减系数mu_all，全部使用numpy和numba并行加速。
    正确的相机方向由rotation_matrix_y(rotation_deg)和[0,0,-1]生成。
    返回mu_all矩阵用于后续任意proj_img校正。
    """
    h, w = depth_map.shape
    nx, ny, nz = mu_3d.shape
    mu_all = np.zeros((h, w), dtype=np.float32)

    direction = np.matmul(
        rotation_matrix_y(rotation_deg), np.array([0, 0, -1], dtype=np.float32)
    )
    # 计算世界坐标
    for ix in prange(h):
        for iy in prange(w):
            x0 = (ix - h / 2 + 0.5) * voxel_size_mm
            y0 = (iy - w / 2 + 0.5) * voxel_size_mm
            z0 = 0.0
            start = np.array([x0, y0, z0], dtype=np.float32)
            dep = float(depth_map[ix, iy])
            raysum = 0.0
            steps, lengths = dda_voxel_tracing_numba(
                start, direction, mu_3d.shape, dep, voxel_size_mm
            )
            for k in range(len(steps)):
                idx = steps[k]
                l = lengths[k]
                raysum += float(mu_3d[idx[0], idx[1], idx[2]]) * l
            if dep >= z_virtual:
                raysum += mu_default * (dep - z_virtual)
            else:
                raysum -= mu_default * (z_virtual - dep)
            mu_all[ix, iy] = raysum
    return mu_all


def attenuation_projection_correction_numpy(proj_img, mu_all):
    """
    用预计算mu_all矩阵对任意proj_img做校正。全部numpy实现。
    """
    proj_img = np.asarray(proj_img, dtype=np.float32)
    mu_all = np.asarray(mu_all, dtype=np.float32)
    return proj_img * np.exp(mu_all)
