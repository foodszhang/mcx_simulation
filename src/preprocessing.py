import os
import numpy as np


def preprocess_voxel_blocks(
    voxel_grid_path,
    save_dir,
    x_range,
    y_range,
    z_range,
    block_size=(32, 32, 32),
    include_empty=False,
):
    """
    预处理：将体素网格按块划分并保存

    参数：
        voxel_grid_path: 原始体素网格npy文件路径
        save_dir: 块文件保存目录
        x_range, y_range, z_range: 可行域范围 [min, max]（体素索引）
        block_size: 块大小 (bx, by, bz)
        include_empty: 是否包含空体素
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载体素网格
    voxel_grid = np.load(voxel_grid_path)
    X, Y, Z = voxel_grid.shape

    # 裁剪可行域到体素网格范围内
    x_min, x_max = max(0, int(x_range[0])), min(X, int(x_range[1]))
    y_min, y_max = max(0, int(y_range[0])), min(Y, int(y_range[1]))
    z_min, z_max = max(0, int(z_range[0])), min(Z, int(z_range[1]))

    bx, by, bz = block_size

    # 计算各轴分块数量
    num_blocks_x = max(1, (x_max - x_min + bx - 1) // bx)
    num_blocks_y = max(1, (y_max - y_min + by - 1) // by)
    num_blocks_z = max(1, (z_max - z_min + bz - 1) // bz)

    total_blocks = num_blocks_x * num_blocks_y * num_blocks_z
    print(f"总块数: {total_blocks}，保存路径: {save_dir}")

    # 遍历所有块并保存
    block_idx = 0
    for i in range(num_blocks_x):
        for j in range(num_blocks_y):
            for k in range(num_blocks_z):
                # 计算当前块的坐标范围
                x_start = x_min + i * bx
                x_end = min(x_start + bx, x_max)
                y_start = y_min + j * by
                y_end = min(y_start + by, y_max)
                z_start = z_min + k * bz
                z_end = min(z_start + bz, z_max)

                # 提取块内体素值
                block_voxels = voxel_grid[x_start:x_end, y_start:y_end, z_start:z_end]

                # 生成块内体素的全局坐标
                x_coords = np.arange(x_start, x_end)[:, None, None]
                y_coords = np.arange(y_start, y_end)[None, :, None]
                z_coords = np.arange(z_start, z_end)[None, None, :]

                # 扩展为网格坐标并展平
                coords = np.stack(
                    [
                        x_coords.repeat(y_end - y_start, axis=1).repeat(
                            z_end - z_start, axis=2
                        ),
                        y_coords.repeat(x_end - x_start, axis=0).repeat(
                            z_end - z_start, axis=2
                        ),
                        z_coords.repeat(x_end - x_start, axis=0).repeat(
                            y_end - y_start, axis=1
                        ),
                    ],
                    axis=-1,
                ).reshape(-1, 3)

                values = block_voxels.flatten()

                # 过滤空体素
                if not include_empty:
                    mask = values != 0
                    coords = coords[mask]
                    values = values[mask]

                # 保存块数据（坐标+值+元信息）
                save_path = os.path.join(save_dir, f"block_{block_idx}.npz")
                np.savez(
                    save_path,
                    coords=coords,
                    values=values,
                    x_range=(x_start, x_end),
                    y_range=(y_start, y_end),
                    z_range=(z_start, z_end),
                )

                block_idx += 1
                if block_idx % 100 == 0:
                    print(f"已处理 {block_idx}/{total_blocks} 块")

    print("预处理完成！")


# 预处理示例
if __name__ == "__main__":
    preprocess_voxel_blocks(
        voxel_grid_path="./two_source_train/volume_brain.npy",  # 原始体素网格
        save_dir="preprocessed_blocks",  # 块保存目录
        x_range=[40, 120],  # 可行域x范围
        y_range=[20, 140],  # 可行域y范围
        z_range=[80, 160],  # 可行域z范围
        block_size=(32, 32, 32),  # 块大小
        include_empty=True,  # 过滤空体素
    )
