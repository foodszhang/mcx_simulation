"""
核心逻辑说明（医学/生物成像工程标准注释）：

本模块主要实现三维光场数据在体素网格内（I[x, y, z]）的多视角投影仿真。
- 默认主视角为 z 轴（从大到小）, 即相机由物体顶部朝向底部观察。
- 投影时以 Y 轴（穿过体素网格中心）为旋转轴, 通过旋转 rotation_deg（角度）获得不同侧视角。
- 对每一个 rotation_deg，均以固定的相机与体素网格中心距离进行投影。
- 投影操作时, 对于每个投影射线（像素方向），在体素网格中沿相机方向遍历，取第一个遇到的 I[x, y, z] ≠ 0 的点作为投影结果。
- 该逻辑贴合医学荧光成像、体积数据科学分析需求, 便于多角度采集与生物数据分析。

rotation_deg: 控制当前侧视角（0°为主视角，正方向为顺时针绕Y轴旋转）。
相机方向：由rotation_deg决定，主视角为[0, 0, -1]，旋转后自动变换。

所有关键变量、主流程、核心算法均有详细中文注释，方便医学工程团队和深度学习场景长期维护与复用。

深度/位置/探测参数可灵活设置，直接用于医学仿真或AI数据生成。
"""

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@njit(cache=True)
def rotation_matrix_y(angle_deg):
    """绕Y轴旋转矩阵"""
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    R = np.array([[cos_a, 0, sin_a], [0, 1.0, 0], [-sin_a, 0, cos_a]], dtype=np.float32)
    return R


@njit(cache=True)
def world_to_camera_coords_batch_matrix(points, rotation_deg, camera_distance):
    """
    使用矩阵运算批量将世界坐标系中的点转换到相机坐标系

    参数:
    points: 批量点 [B, N, 3] 或 [N, 3]
    rotation_deg: 旋转角度
    camera_distance: 相机距离

    返回:
    camera_coords: 相机坐标系中的点 [B, N, 3] 或 [N, 3]
    """
    R = rotation_matrix_y(rotation_deg)

    if points.ndim == 3:
        B, N, _ = points.shape
        # 重塑为 [B*N, 3] 进行矩阵乘法
        points_flat = points.reshape(B * N, 3)
        camera_coords_flat = points_flat @ R.T  # 矩阵乘法
        # 重塑回 [B, N, 3]
        camera_coords = camera_coords_flat.reshape(B, N, 3)
    else:  # points.ndim == 2
        camera_coords = points @ R.T  # 直接矩阵乘法

    # 修正深度计算
    camera_coords[..., 2] = camera_distance - camera_coords[..., 2]

    return camera_coords


@njit(cache=True)
def project_points_to_camera_batch_matrix(
    points, rotation_deg, camera_distance, detector_size
):
    """
    使用矩阵运算批量投影点

    参数:
    points: 批量点 [B, N, 3] 或 [N, 3]
    rotation_deg: 旋转角度
    camera_distance: 相机距离
    detector_size: 探测器尺寸 (width, height)

    返回:
    projections: 投影坐标 [B, N, 2] 或 [N, 2]
    depths: 深度信息 [B, N, 2] 或 [N, 2] (深度值, 可见性标志)
    """
    # 使用矩阵运算转换坐标
    camera_coords = world_to_camera_coords_batch_matrix(
        points, rotation_deg, camera_distance
    )

    width, height = detector_size

    # 提取UV坐标和深度
    if points.ndim == 3:
        B, N, _ = points.shape
        u = camera_coords[:, :, 0]  # [B, N]
        v = camera_coords[:, :, 1]  # [B, N]
        depth_vals = camera_coords[:, :, 2]  # [B, N]

        # 计算可见性 (使用向量化操作)
        visible = (
            (np.abs(u) <= width / 2) & (np.abs(v) <= height / 2) & (depth_vals > 0)
        )

        # 构建投影坐标 [B, N, 2]
        projections = np.zeros((B, N, 2), dtype=np.float32)
        projections[:, :, 0] = u
        projections[:, :, 1] = v

        # 构建深度信息 [B, N, 2]
        depths = np.zeros((B, N, 2), dtype=np.float32)
        depths[:, :, 0] = depth_vals
        depths[:, :, 1] = visible.astype(np.float32)

    else:  # points.ndim == 2
        N, _ = points.shape
        u = camera_coords[:, 0]  # [N]
        v = camera_coords[:, 1]  # [N]
        depth_vals = camera_coords[:, 2]  # [N]

        # 计算可见性
        visible = (
            (np.abs(u) <= width / 2) & (np.abs(v) <= height / 2) & (depth_vals > 0)
        )

        # 构建投影坐标 [N, 2]
        projections = np.zeros((N, 2), dtype=np.float32)
        projections[:, 0] = u
        projections[:, 1] = v

        # 构建深度信息 [N, 2]
        depths = np.zeros((N, 2), dtype=np.float32)
        depths[:, 0] = depth_vals
        depths[:, 1] = visible.astype(np.float32)

    return projections, depths


@njit(cache=True)
def generate_projection_view_matrix(
    voxel_data, rotation_deg, camera_distance, detector_size, detector_resolution
):
    """
    使用矩阵运算优化的投影视图生成。
    注意：该函数不做任何深度过滤，完全保留原始所有可见点，只在展示阶段做后处理。
    """
    width_pixels, height_pixels = detector_resolution
    width_phys, height_phys = detector_size

    projection = np.zeros((height_pixels, width_pixels), dtype=np.float32)
    depth_map = np.full((height_pixels, width_pixels), np.inf, dtype=np.float32)

    pixel_to_phys_x = width_phys / width_pixels
    pixel_to_phys_y = height_phys / height_pixels

    nx, ny, nz = voxel_data.shape

    # 收集所有非零体素的位置
    nonzero_indices = []
    values = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if voxel_data[i, j, k] != 0:
                    x = i - nx / 2 + 0.5
                    y = j - ny / 2 + 0.5
                    z = k - nz / 2 + 0.5
                    nonzero_indices.append([x, y, z])
                    values.append(voxel_data[i, j, k])

    if len(nonzero_indices) == 0:
        return projection, depth_map

    # 转换为numpy数组进行批量处理
    points = np.array(nonzero_indices, dtype=np.float32)
    values_array = np.array(values, dtype=np.float32)

    projections, depths = project_points_to_camera_batch_matrix(
        points, rotation_deg, camera_distance, detector_size
    )

    # 不做任何深度过滤，只保留最浅点
    for idx in range(len(points)):
        u, v = projections[idx, 0], projections[idx, 1]
        depth_val = depths[idx, 0]
        visible = depths[idx, 1] > 0.5

        if visible:
            # 转换到像素坐标
            pixel_u = int((u + width_phys / 2) / pixel_to_phys_x)
            pixel_v = int((v + height_phys / 2) / pixel_to_phys_y)

            # 修正上下颠倒，如果需要可解开注释
            # pixel_v = height_pixels - 1 - pixel_v

            if 0 <= pixel_u < width_pixels and 0 <= pixel_v < height_pixels:
                # 只保留该像素最浅点
                if depth_val < depth_map[pixel_u, pixel_v]:
                    depth_map[pixel_u, pixel_v] = depth_val
                    projection[pixel_u, pixel_v] = values_array[idx]

    return projection, depth_map


# 更新VolumeProjector类以使用矩阵运算版本
class VolumeProjector:
    """体积数据投影器 - 使用矩阵运算优化"""

    def __init__(
        self, camera_distance=40, detector_size=(25, 25), detector_resolution=(200, 200)
    ):
        self.camera_distance = camera_distance
        self.detector_size = np.array(detector_size, dtype=np.float32)
        self.detector_resolution = np.array(detector_resolution, dtype=np.int32)

    def project_volume(self, volume_data, view_angles=None):
        """使用矩阵运算优化的体积投影，支持最大深度阈值控制"""
        if view_angles is None:
            view_angles = [0, 30, 60, 90, 120, 150, 180]

        projections = []
        depth_maps = []
        angles_list = []

        print(f"开始生成 {len(view_angles)} 个视角的投影 (矩阵运算优化)...")
        for angle in view_angles:
            print(f"  生成 {angle}° 视角投影...")
            proj, depth = generate_projection_view_matrix(
                volume_data,
                angle,
                self.camera_distance,
                self.detector_size,
                self.detector_resolution,
            )
            projections.append(proj)
            depth_maps.append(depth)
            angles_list.append(angle)

        return projections, depth_maps, angles_list

    def visualize_projections(self, volume_data, view_angles=None, figsize=(20, 8)):
        """ """
        if view_angles is None:
            view_angles = [0, 30, 60, 90, 120, 150, 180]

        projections, depth_maps, angles_list = self.project_volume(
            volume_data, view_angles
        )

        n_views = len(view_angles)
        fig, axes = plt.subplots(2, n_views, figsize=figsize)

        if n_views == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        for idx, angle in enumerate(angles_list):
            projection = projections[idx]
            depth_map = depth_maps[idx]

            # 只在展示阶段做深度分位过滤，科学可溯源
            # 获取有效深度点（去除inf），并计算80%分位点
            valid_depths = depth_map[depth_map != np.inf]
            if valid_depths.size > 0:
                depth_thr = np.percentile(valid_depths, 80)
            else:
                depth_thr = np.inf
            # 生成掩码：只显示深度小于该阈值的点
            mask = (depth_map < depth_thr) & (depth_map != np.inf)

            # 应用掩码：显示80%最浅的点
            filtered_projection = np.zeros_like(projection)
            filtered_projection[mask] = projection[mask]
            filtered_depth = np.zeros_like(depth_map)
            filtered_depth[mask] = depth_map[mask]

            if n_views == 1:
                ax1 = axes[0, 0]
                ax2 = axes[1, 0]
            else:
                ax1 = axes[0, idx]
                ax2 = axes[1, idx]

            extent = np.array(
                [
                    -self.detector_size[0] / 2,
                    self.detector_size[0] / 2,
                    -self.detector_size[1] / 2,
                    self.detector_size[1] / 2,
                ],
                dtype=np.float32,
            )

            # 展示处理后的投影图和深度图，贴合医学科研场景
            im1 = ax1.imshow(filtered_projection, cmap="hot", extent=extent)
            ax1.set_title(f"投影视图 {angle}° (仅80%浅层像素)")
            plt.colorbar(im1, ax=ax1, fraction=0.046)

            im2 = ax2.imshow(filtered_depth, cmap="viridis", extent=extent)
            ax2.set_title(f"深度图 {angle}° (小于P80)")
            plt.colorbar(im2, ax=ax2, fraction=0.046)

        plt.tight_layout()
        plt.show()

        projections_dict = {}
        depth_maps_dict = {}
        for i, angle in enumerate(angles_list):
            projections_dict[angle] = projections[i]
            depth_maps_dict[angle] = depth_maps[i]

        return projections_dict, depth_maps_dict


def analyze_volume(
    volume_data,
    view_angles=None,
    camera_distance=40,
    detector_size=(25, 25),
    detector_resolution=(200, 200),
):
    """
    分析任意体积数据的多角度投影

    参数:
    volume_data: 3D numpy数组
    view_angles: 视角列表
    camera_distance: 相机距离
    detector_size: 探测板尺寸
    detector_resolution: 探测板分辨率
    """
    if view_angles is None:
        view_angles = [0, 30, 60, 90, 120, 150, 180]

    projector = VolumeProjector(camera_distance, detector_size, detector_resolution)

    print(f"体积数据分析")
    print(f"输入数据形状: {volume_data.shape}")
    print(f"数据范围: [{volume_data.min():.3f}, {volume_data.max():.3f}]")
    print(f"非零元素: {np.count_nonzero(volume_data)} / {volume_data.size}")

    # 生成投影
    projections, depth_maps = projector.visualize_projections(volume_data, view_angles)

    return projections, depth_maps


if __name__ == "__main__":
    # 预编译Numba函数
    print("预编译Numba函数...")
    import jdata as jd

    full_data = jd.loadjd("./20251106/0/no_0.jnii")
    if len(full_data["NIFTIData"].shape) == 3:
        flux = full_data["NIFTIData"][:, :, :]
    else:
        flux = full_data["NIFTIData"][:, :, :, 0, 0]

    # 示例：如何使用自定义数据
    print("\n" + "=" * 50)
    print("自定义数据示例")
    print("=" * 50)

    # 创建自定义体积数据

    # 添加一些结构

    # 分析自定义数据
    # flux = np.where(flux > 1, np.log(flux), 0)
    analyze_volume(
        flux,
        view_angles=[-90, -60, -30, 0, 30, 60, 90],
        camera_distance=200,
        detector_resolution=(256, 256),
        detector_size=(256, 300),
    )
