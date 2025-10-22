import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def show_heatmaps_original_size(npz_file_path):
    """
    读取npz文件中的d1-d4灰度图像，按原图尺寸比例展示热力图（不缩放）
    """
    try:
        # 读取npz文件
        data = np.load(npz_file_path)
        required_keys = ["d1", "d2", "d3", "d4"]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"npz文件中缺少以下键: {', '.join(missing_keys)}")

        # 获取所有图像的原始尺寸
        images = {}
        for key in required_keys:
            img = data[key]
            if len(img.shape) != 2:
                raise ValueError(f"{key}不是二维灰度图像，形状为{img.shape}")
            images[key] = img

        # 计算每个图像的宽高比，用于设置子图尺寸
        aspect_ratios = {k: img.shape[1] / img.shape[0] for k, img in images.items()}

        # 创建图形，根据图像尺寸动态调整布局
        plt.figure(figsize=(16, 12))

        # 使用GridSpec并设置宽度比例，保持原始尺寸比例
        gs = gridspec.GridSpec(
            2,
            2,
            width_ratios=[aspect_ratios["d1"], aspect_ratios["d2"]],
            height_ratios=[1 / aspect_ratios["d1"], 1 / aspect_ratios["d3"]],
            hspace=0.3,
            wspace=0.3,
        )

        # 绘制每个热力图（使用aspect='equal'保持原始比例）
        for i, key in enumerate(required_keys):
            img_data = images[key]
            ax = plt.subplot(gs[i])

            # 关键设置：aspect='equal'确保不缩放，保持原始尺寸比例
            im = ax.imshow(img_data, cmap="viridis", aspect="equal")

            ax.set_title(
                f"热力图: {key} (尺寸: {img_data.shape[1]}x{img_data.shape[0]})",
                fontsize=10,
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # 显示坐标轴刻度以体现实际尺寸比例
            ax.set_xticks(np.linspace(0, img_data.shape[1] - 1, 5).astype(int))
            ax.set_yticks(np.linspace(0, img_data.shape[0] - 1, 5).astype(int))
            ax.tick_params(axis="both", which="major", labelsize=8)

        plt.suptitle("按原始尺寸比例展示的热力图", fontsize=16)
        plt.show()

    except FileNotFoundError:
        print(f"错误: 找不到文件 {npz_file_path}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        if "data" in locals():
            data.close()


# 使用示例
if __name__ == "__main__":
    npz_file_path = "./20251019/100/proj.npz"  # 替换为你的npz文件路径
    show_heatmaps_original_size(npz_file_path)
