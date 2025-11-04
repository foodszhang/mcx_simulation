"""
坐标系变换测试脚本
用于验证 transform_flux_for_projection 函数
以及不同变换选项对投影结果的影响
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from gen_mul_projection import generate_projection_view_matrix


def transform_flux_for_projection(flux, transform_option="option1"):
    """
    将原始flux变换为符合generate_projection_view_matrix坐标系的格式

    generate_projection_view_matrix期望的坐标系：
    - 输入形状：(nx, ny, nz)
    - voxel_data[i, j, k] 对应世界坐标 (x, y, z)
    - 其中：i→X轴，j→Y轴，k→Z轴
    - 旋转轴：Y轴（绕Y轴旋转）

    参数：
    flux: 原始3D数据
    transform_option: 变换方案
        - "option1": 直接使用（如果原始数据已经是正确的轴顺序）
        - "option2": transpose((2,1,0)) - 反转所有轴顺序（MATLAB常用）
        - "option3": transpose((1,0,2)) - 交换X和Y轴，保持Z轴
        - "option4": transpose((2,0,1)) - MCX标准格式调整
        - "option5": 带翻转的变换
    """
    print(f"原始flux形状: {flux.shape}")

    if transform_option == "option1":
        # 直接使用，无变换
        transformed = flux.copy()
        print("使用 option1: 直接使用原始数据（无变换）")

    elif transform_option == "option2":
        # 反转所有轴顺序（MATLAB v7.3常需要此操作）
        transformed = np.transpose(flux, (2, 1, 0))
        print("使用 option2: transpose((2,1,0)) - 反转轴顺序")

    elif transform_option == "option3":
        # 交换X和Y轴
        transformed = np.transpose(flux, (1, 0, 2))
        print("使用 option3: transpose((1,0,2)) - 交换X和Y轴")

    elif transform_option == "option4":
        # MCX标准格式调整
        transformed = np.transpose(flux, (2, 0, 1))
        print("使用 option4: transpose((2,0,1)) - MCX标准调整")

    elif transform_option == "option5":
        # 带翻转的变换：先transpose后flip Z轴
        transformed = np.transpose(flux, (2, 1, 0))
        transformed = np.flip(transformed, axis=2)
        print("使用 option5: transpose((2,1,0)) + flip Z轴")

    else:
        raise ValueError(f"未知的变换选项: {transform_option}")

    print(f"变换后flux形状: {transformed.shape}")
    return transformed


def test_single_option(flux_data, option_name, test_angles=[-90, 0, 90]):
    """
    测试单个变换选项的投影效果
    """
    print(f"\n{'='*60}")
    print(f"测试变换选项: {option_name}")
    print(f"{'='*60}")

    # 应用变换
    transformed = transform_flux_for_projection(flux_data, option_name)

    # 显示数据统计
    print(f"数据统计:")
    print(f"  形状: {transformed.shape}")
    print(f"  最小值: {transformed.min():.6f}")
    print(f"  最大值: {transformed.max():.6f}")
    print(f"  平均值: {transformed.mean():.6f}")
    print(f"  非零元素数: {np.count_nonzero(transformed)} / {transformed.size}")

    # 显示中心切片
    nx, ny, nz = transformed.shape
    print(f"\n中心切片数据:")
    print(f"  Z轴中心切片 (Z={nz//2}): 形状 {transformed[:, :, nz//2].shape}, 非零元素: {np.count_nonzero(transformed[:, :, nz//2])}")
    print(f"  Y轴中心切片 (Y={ny//2}): 形状 {transformed[:, ny//2, :].shape}, 非零元素: {np.count_nonzero(transformed[:, ny//2, :])}")
    print(f"  X轴中心切片 (X={nx//2}): 形状 {transformed[nx//2, :, :].shape}, 非零元素: {np.count_nonzero(transformed[nx//2, :, :])}")

    # 生成投影测试
    print(f"\n生成投影...")
    projections = {}
    for angle in test_angles:
        proj, _ = generate_projection_view_matrix(
            transformed,
            angle,
            200,
            (256, 256),
            (256, 256),
        )
        projections[angle] = proj
        print(f"  角度 {angle}°: 投影形状 {proj.shape}, 非零元素: {np.count_nonzero(proj)}, 最大值: {proj.max():.6f}")

    return transformed, projections


if __name__ == "__main__":
    print("加载数据...")
    with h5py.File("./1.mat", "r") as mat:
        print(f"可用的数据集: {list(mat.keys())}")
        flux_injure = np.array(mat["CWfluencem_injure_2"])

    print(f"\n原始数据形状: {flux_injure.shape}")
    print(f"数据统计: min={flux_injure.min():.6f}, max={flux_injure.max():.6f}, mean={flux_injure.mean():.6f}")

    # 测试所有变换选项
    all_results = {}
    test_angles = [-90, -30, 0, 30, 90]

    for option in ["option1", "option2", "option3", "option4", "option5"]:
        try:
            transformed, projections = test_single_option(flux_injure, option, test_angles)
            all_results[option] = {
                "transformed": transformed,
                "projections": projections
            }
        except Exception as e:
            print(f"错误: {e}")

    # 可视化比较
    print(f"\n{'='*60}")
    print("生成可视化对比...")
    print(f"{'='*60}")

    fig, axes = plt.subplots(len(all_results), len(test_angles), figsize=(15, 12))

    for row_idx, (option, data) in enumerate(all_results.items()):
        projections = data["projections"]
        for col_idx, angle in enumerate(test_angles):
            ax = axes[row_idx, col_idx]
            proj = projections[angle]

            if proj.max() > 0:
                im = ax.imshow(proj, cmap="hot")
            else:
                ax.imshow(proj, cmap="hot")

            ax.set_title(f"{option} @ {angle}°")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            if col_idx == len(test_angles) - 1:
                plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("./coordinate_transform_comparison.png", dpi=100)
    print("可视化已保存到: ./coordinate_transform_comparison.png")

    # 推荐选项
    print(f"\n{'='*60}")
    print("选择建议:")
    print(f"{'='*60}")
    print("根据投影结果的非零元素数量和投影质量，请选择最合适的变换选项。")
    print("通常 MATLAB v7.3 数据推荐使用 option2 或 option4。")
    print("请查看生成的图像对比来判断哪个选项最符合您的预期。")

