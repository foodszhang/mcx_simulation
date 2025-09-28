import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
import random


# 生成示例三维数组（模拟分割结果）
def generate_sample_3d_array(size=15):
    """生成一个模拟分割结果的三维数组，值从外到内递增"""
    arr = np.zeros((size, size, size), dtype=int)

    # 生成一个球体作为基础形状
    center = size // 2
    max_radius = center - 1

    for x in range(size):
        for y in range(size):
            for z in range(size):
                # 计算到中心的距离
                dx = x - center
                dy = y - center
                dz = z - center
                dist = np.sqrt(dx**2 + dy**2 + dz**2)

                # 根据距离分配值（距离越远值越小，越靠外）
                if dist < max_radius:
                    # 值的范围从1到5，越靠内值越大
                    value = int((1 - dist / max_radius) * 4) + 1
                    arr[x, y, z] = value

    return arr


def visualize_3d_array(arr, title="3D数组分割结果可视化"):
    """
    可视化三维数组，值越小（越靠外）颜色越浅、透明度越高
    arr: 要可视化的三维数组
    title: 图表标题
    """
    # 确保数组是三维的
    if len(arr.shape) != 3:
        raise ValueError("输入必须是三维数组")

    # 获取数组的非零元素坐标和值
    x, y, z = np.where(arr > 0)
    values = arr[x, y, z]

    # 如果没有非零元素，显示空图
    if len(values) == 0:
        print("数组中没有非零元素可显示")
        return

    # 确定值的范围用于归一化
    min_val = np.min(values)
    max_val = np.max(values)

    # 创建图形和3D轴
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 归一化值用于颜色和透明度映射
    norm = Normalize(vmin=min_val, vmax=max_val)

    # 使用jet色彩映射，值越小颜色越浅
    # 反转映射，使值越小颜色越浅（红色→黄色→蓝色→深蓝色）
    cmap = cm.get_cmap("jet")

    # 颜色映射：值越小颜色越浅
    colors = cmap(norm(values))

    # 透明度映射：值越小透明度越高（0.2到0.8之间）
    # 注意：这里使用1 - norm(values)使值小的透明度高
    alphas = 0.2 + (1 - norm(values)) * 0.6
    colors[:, 3] = alphas  # 设置RGBA中的alpha通道

    # 绘制3D散点图
    scatter = ax.scatter(
        x,
        y,
        z,
        c=colors,
        marker="o",
        s=10,  # 点的大小
    )

    # 设置标题和轴标签
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("X轴", fontsize=12)
    ax.set_ylabel("Y轴", fontsize=12)
    ax.set_zlabel("Z轴", fontsize=12)

    # 设置轴范围
    ax.set_xlim(0, arr.shape[0] - 1)
    ax.set_ylim(0, arr.shape[1] - 1)
    ax.set_zlim(0, arr.shape[2] - 1)

    # 添加颜色条显示值的对应关系
    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array(values)
    cbar = fig.colorbar(m, ax=ax, pad=0.1)
    cbar.set_label("分割值（值越小越靠外）", fontsize=10)

    # 调整视角
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.show()


# 扩展：生成带有多个区域的复杂分割结果
def generate_complex_segmentation(size=20):
    """生成更复杂的分割结果，包含多个区域"""
    arr = np.zeros((size, size, size), dtype=int)
    center = size // 2

    # 生成中心球体区域
    max_radius = size // 3
    for x in range(size):
        for y in range(size):
            for z in range(size):
                dx = x - center
                dy = y - center
                dz = z - center
                dist = np.sqrt(dx**2 + dy**2 + dz**2)

                if dist < max_radius:
                    value = int((1 - dist / max_radius) * 3) + 1  # 值1-4
                    arr[x, y, z] = value

    # 生成一个附加的椭球区域
    for x in range(size):
        for y in range(size):
            for z in range(size):
                dx = (x - (center + 5)) / 8
                dy = (y - center) / 5
                dz = (z - center) / 6

                if dx**2 + dy**2 + dz**2 < 1:
                    # 这个区域的值从5开始，与中心区域区分
                    value = 5 + int((1 - (dx**2 + dy**2 + dz**2)) * 2)  # 值5-7
                    arr[x, y, z] = value

    return arr


# 示例用法
if __name__ == "__main__":
    # 生成简单的示例三维数组
    simple_array = generate_sample_3d_array(size=15)
    print(f"简单示例数组形状: {simple_array.shape}")
    visualize_3d_array(simple_array, title="简单三维分割结果可视化")

    # 生成并显示复杂分割结果
    complex_array = generate_complex_segmentation(size=20)
    print(f"复杂示例数组形状: {complex_array.shape}")
    visualize_3d_array(complex_array, title="复杂三维分割结果可视化")
