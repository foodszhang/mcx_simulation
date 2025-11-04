import numpy as np
import matplotlib.pyplot as plt
import os


def display_npz_heatmaps(npz_path, save_dir=None, figsize=(12, 8), cmap="hot"):
    """
    读取NPZ文件并将多个灰度图用热力图方式展示

    参数:
    - npz_path: NPZ文件路径
    - save_dir: 保存图片的目录，如果为None则不保存
    - figsize: 图像大小
    - cmap: 颜色映射，可选 'hot', 'jet', 'viridis', 'plasma' 等
    """
    try:
        # 加载NPZ文件
        data = np.load(npz_path)
        print(f"NPZ文件中的数组: {list(data.keys())}")

        # 创建保存目录
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # 遍历所有数组
        for key in data.keys():
            array = data[key]
            print(f"数组 '{key}' 的形状: {array.shape}")

            # 处理不同维度的数组
            if array.ndim == 2:
                # 单个灰度图
                display_single_heatmap(array, key, save_dir, figsize, cmap)

            elif array.ndim == 3:
                # 多个灰度图
                display_multiple_heatmaps(array, key, save_dir, figsize, cmap)

            elif array.ndim == 1:
                print(f"跳过一维数组 '{key}'")

            else:
                print(f"无法处理 {array.ndim} 维数组 '{key}'")

    except Exception as e:
        print(f"读取NPZ文件时出错: {e}")


def display_single_heatmap(array, title, save_dir=None, figsize=(8, 6), cmap="hot"):
    """显示单个热力图"""
    plt.figure(figsize=figsize)

    plt.imshow(array, cmap=cmap)
    plt.colorbar(label="强度值")
    plt.title(f"热力图 - {title}")
    plt.axis("off")  # 隐藏坐标轴

    # 添加统计信息
    stats_text = (
        f"Min: {array.min():.2f}\nMax: {array.max():.2f}\nMean: {array.mean():.2f}"
    )
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    if save_dir is not None:
        filename = os.path.join(save_dir, f"heatmap_{title}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"已保存: {filename}")

    plt.tight_layout()
    plt.show()


def display_multiple_heatmaps(
    array, title, save_dir=None, figsize=(15, 10), cmap="hot", max_display=16
):
    """显示多个热力图"""
    num_images = array.shape[0]

    # 限制显示数量，避免太多图像
    if num_images > max_display:
        print(f"数组 '{title}' 包含 {num_images} 个图像，只显示前 {max_display} 个")
        array = array[:max_display]
        num_images = max_display

    # 计算网格布局
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # 如果只有一行或一列，确保axes是二维数组
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < num_images:
                ax = axes[i, j]
                im = ax.imshow(array[idx], cmap=cmap)
                ax.set_title(f"{title}[{idx}]")
                ax.axis("off")

                # 为每个子图添加颜色条
                plt.colorbar(im, ax=ax, shrink=0.8)
            else:
                axes[i, j].axis("off")

    plt.suptitle(f"多热力图展示 - {title} (共{num_images}个图像)")
    plt.tight_layout()

    if save_dir is not None:
        filename = os.path.join(save_dir, f"multi_heatmap_{title}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"已保存: {filename}")

    plt.show()


# 使用示例
if __name__ == "__main__":
    # 示例1: 基本用法
    npz_path = "./one_source_val/3/proj.npz"
    display_npz_heatmaps(npz_path, cmap="hot")
    no_npz_path = "./one_source_val/3/no_proj.npz"
    display_npz_heatmaps(no_npz_path, cmap="hot")

    # 示例2: 保存图片并使用不同颜色映射
    # display_npz_heatmaps(npz_path, save_dir='heatmap_results', cmap='viridis')

    # 示例3: 如果你知道NPZ文件中的特定数组名，也可以单独处理
    # data = np.load(npz_path)
    # specific_array = data['array_name']  # 替换为实际的数组名
    # display_single_heatmap(specific_array, '特定数组')
