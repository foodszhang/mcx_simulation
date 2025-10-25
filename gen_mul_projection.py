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
    使用矩阵运算优化的投影视图生成
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

    # 批量投影
    projections, depths = project_points_to_camera_batch_matrix(
        points, rotation_deg, camera_distance, detector_size
    )

    # 处理每个投影点
    for idx in range(len(points)):
        u, v = projections[idx, 0], projections[idx, 1]
        depth_val = depths[idx, 0]
        visible = depths[idx, 1] > 0.5

        if visible:
            # 转换到像素坐标
            pixel_u = int((u + width_phys / 2) / pixel_to_phys_x)
            pixel_v = int((v + height_phys / 2) / pixel_to_phys_y)

            # 修正上下颠倒
            # pixel_v = height_pixels - 1 - pixel_v

            if 0 <= pixel_u < width_pixels and 0 <= pixel_v < height_pixels:
                if depth_val < depth_map[pixel_v, pixel_u]:
                    depth_map[pixel_v, pixel_u] = depth_val
                    projection[pixel_v, pixel_u] = values_array[idx]

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
        """使用矩阵运算优化的体积投影"""
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
        """可视化投影结果"""
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

            im1 = ax1.imshow(projection, cmap="hot", extent=extent)
            ax1.set_title(f"投影视图 {angle}°")
            plt.colorbar(im1, ax=ax1, fraction=0.046)

            depth_display = depth_map.copy()
            depth_display[depth_display == np.inf] = 0
            im2 = ax2.imshow(depth_display, cmap="viridis", extent=extent)
            ax2.set_title(f"深度图 {angle}°")
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

    full_data = jd.loadjd("./20251021/0/0.jnii")
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
        view_angles=[-90, -30, -60, 0, 30, 60, 90],
        camera_distance=200,
        detector_resolution=(256, 256),
        detector_size=(256, 256),
    )
