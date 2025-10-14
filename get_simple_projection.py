import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def get_projections(flux_matrix, tag_matrix):
    """
    从Z轴朝下获取每个(X,Y)位置上第一个tag不为0的点的flux值

    参数:
        flux_matrix: 三维光通量矩阵，形状为(Z, Y, X)
        tag_matrix: 三维标签矩阵，形状需与flux_matrix相同

    返回:
        surface_flux: 二维矩阵，形状为(Y, X)，每个元素为对应位置第一个非零标签点的光通量值
    """
    # 检查输入矩阵形状是否一致
    if flux_matrix.shape != tag_matrix.shape:
        raise ValueError(
            f"flux矩阵形状 {flux_matrix.shape} 与tag矩阵形状 {tag_matrix.shape} 不匹配"
        )

    # 获取矩阵维度 (Z, Y, X)
    x_size, y_size, z_size = flux_matrix.shape

    # 初始化输出的二维矩阵，默认值为0（表示没有找到非零标签点）
    flux_proj = np.zeros((x_size, y_size), dtype=flux_matrix.dtype)

    tag_proj = np.zeros((x_size, y_size), dtype=flux_matrix.dtype)
    # 遍历每个(X,Y)位置
    for y in range(y_size):
        for x in range(x_size):
            # 从Z轴顶部（索引0）向下查找第一个非零标签点
            for z in range(z_size - 1, 0, -1):
                if tag_matrix[x, y, z] != 0:
                    # 找到第一个非零标签点，记录对应的光通量值
                    flux_proj[x, y] = flux_matrix[x, y, z]
                    tag_proj[x, y] = 1
                    break  # 找到后退出当前Z轴循环

    return flux_proj, tag_proj


def get_multi_direction_projections(flux_matrix, tag_matrix):
    """
    从四个方向获取每个位置上第一个tag不为0的点的flux值

    方向包括：Z轴朝上、Z轴朝下、X轴朝上、X轴朝下

    参数:
        flux_matrix: 三维光通量矩阵，形状为(Z, Y, X)
        tag_matrix: 三维标签矩阵，形状需与flux_matrix相同

    返回:
        projections: 字典，包含四个方向的投影结果，每个结果是一个元组(flux_proj, tag_proj)
                     键分别为: 'z_up', 'z_down', 'x_up', 'x_down'
    """
    # 检查输入矩阵形状是否一致
    if flux_matrix.shape != tag_matrix.shape:
        raise ValueError(
            f"flux矩阵形状 {flux_matrix.shape} 与tag矩阵形状 {tag_matrix.shape} 不匹配"
        )

    x_size, y_size, z_size = flux_matrix.shape
    projections = {}

    # 1. Z轴朝下 (从Z=0向Z=z_size-1查找)
    flux_z_down = np.zeros((x_size, y_size), dtype=flux_matrix.dtype)
    tag_z_down = np.zeros((x_size, y_size), dtype=tag_matrix.dtype)

    for y in range(y_size):
        for x in range(x_size):
            for z in range(z_size):  # 从Z轴底部（索引0）向上查找
                if tag_matrix[x, y, z] != 0:
                    flux_z_down[x, y] = flux_matrix[x, y, z]
                    tag_z_down[x, y] = 1
                    break

    projections["d1"] = flux_z_down

    # 2. Z轴朝上 (从Z=z_size-1向Z=0查找)
    flux_z_up = np.zeros((x_size, y_size), dtype=flux_matrix.dtype)
    tag_z_up = np.zeros((x_size, y_size), dtype=tag_matrix.dtype)

    for y in range(y_size):
        for x in range(x_size):
            for z in range(z_size - 1, -1, -1):  # 从Z轴顶部向下查找
                if tag_matrix[x, y, z] != 0:
                    flux_z_up[x, y] = flux_matrix[x, y, z]
                    tag_z_up[x, y] = 1
                    break

    projections["d2"] = flux_z_up
    # 3. X轴朝下 (从X=0向X=x_size-1查找)
    flux_x_down = np.zeros((y_size, z_size), dtype=flux_matrix.dtype)
    tag_x_down = np.zeros((y_size, z_size), dtype=tag_matrix.dtype)

    for z in range(z_size):
        for y in range(y_size):
            for x in range(x_size):  # 从X轴左侧向右查找
                if tag_matrix[x, y, z] != 0:
                    flux_x_down[y, z] = flux_matrix[x, y, z]
                    tag_x_down[y, z] = 1
                    break

    projections["d3"] = flux_x_down
    # 4. X轴朝上 (从X=x_size-1向X=0查找)
    flux_x_up = np.zeros((y_size, z_size), dtype=flux_matrix.dtype)
    tag_x_up = np.zeros((y_size, z_size), dtype=tag_matrix.dtype)

    for z in range(z_size):
        for y in range(y_size):
            for x in range(x_size - 1, -1, -1):  # 从X轴右侧向左查找
                if tag_matrix[x, y, z] != 0:
                    flux_x_up[y, z] = flux_matrix[x, y, z]
                    tag_x_up[y, z] = 1
                    break

    # projections["x_up"] = (flux_x_up, tag_x_up)
    projections["d4"] = flux_x_up
    return projections


if __name__ == "__main__":
    tag_mat = np.fromfile("./20251010/volume_brain.bin", dtype=np.uint8).reshape(
        [180, 300, 208]
    )
    # tag_mat = np.transpose(tag_mat, (2, 1, 0))
    import jdata as jd

    full_data = jd.loadjd("./20251010/0/0.jnii")
    if len(full_data["NIFTIData"].shape) == 3:
        flux = full_data["NIFTIData"][:, :, :]
    else:
        flux = full_data["NIFTIData"][:, :, :, 0, 0]
    # print("666666", flux.flags)
    projections = get_multi_direction_projections(flux, tag_mat)
    flux_proj = projections["d4"]
    # flux_proj, tag_proj = get_projections(flux, tag_mat)
    print("6666666", flux_proj.shape)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax_flux = fig.add_subplot(121)
    ax_tag = fig.add_subplot(122)

    # 可以自定义颜色映射（可选）
    cmap_name = "custom"
    if cmap_name == "custom":
        colors = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)
    else:
        cmap = plt.get_cmap(cmap_name)

    # 绘制二维矩阵
    im = ax_flux.imshow(flux_proj, cmap=cmap, interpolation="bilinear", origin="upper")
    flux = np.where(flux > 0, np.log(flux), 0)
    log_projections = get_multi_direction_projections(flux, tag_mat)
    log_flux_proj = log_projections["d4"]
    #
    # log_flux_proj, tag_proj = get_projections(flux, tag_mat)
    #

    # pim = ax_tag.imshow(
    # tag_proj, vmin=0, vmax=1, interpolation="bilinear", origin="upper"
    # )
    pim = ax_tag.imshow(log_flux_proj, interpolation="bilinear", origin="upper")
    # 设置标题和坐标轴标签
    # ax.set_title(title, fontsize=14)
    ax_flux.set_xlabel("X轴", fontsize=12)
    ax_flux.set_ylabel("Y轴", fontsize=12)

    # cbar = fig.colorbar(im, ax=ax_flux)
    # cbar.set_label("光通量速率", fontsize=12)

    # 调整布局
    plt.tight_layout()

    # 显示图像
    plt.show()
