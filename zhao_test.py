import numpy as np
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import rcParams

# 设置字体
font_candidates = ["Arial", "Times New Roman", "Calibri", "DejaVu Sans", "SimHei"]
rcParams["font.family"] = font_candidates

rcParams["axes.unicode_minus"] = False

# 使用h5py读取MATLAB v7.3文件


def show_flux(flux):
    dim_x, dim_y, dim_z = flux.shape

    # 创建图形和子图，布局类似CT图像显示
    fig = plt.figure(figsize=(10, 10))  # 使用正方形画布

    # 主视图：Z轴切面 (X-Y平面) - 固定Z，显示所有X和Y
    ax_z = fig.add_subplot(221)
    # 侧视图：Y轴切面 (X-Z平面) - 固定Y，显示所有X和Z
    ax_y = fig.add_subplot(222)
    # 俯视图：X轴切面 (Y-Z平面) - 固定X，显示所有Y和Z
    ax_x = fig.add_subplot(223)

    # 调整布局，为滑块留出空间
    plt.subplots_adjust(left=0.15, bottom=0.25, right=0.85, top=0.9)

    # 初始切片位置（中心位置）
    init_x = dim_x // 2
    init_y = dim_y // 2
    init_z = dim_z // 2

    # 显示初始切片（设置aspect='equal'确保正方形显示）
    im_z = ax_z.imshow(
        flux[:, :, init_z], cmap="viridis", aspect="equal", origin="lower"
    )
    im_y = ax_y.imshow(
        flux[:, init_y, :].T, cmap="viridis", aspect="equal", origin="lower"
    )
    im_x = ax_x.imshow(
        flux[init_x, :, :].T, cmap="viridis", aspect="equal", origin="lower"
    )

    # 添加颜色条
    cbar = fig.colorbar(
        im_z,
        ax=[ax_z, ax_y, ax_x],
        orientation="horizontal",
        fraction=0.05,
        pad=0.1,
        label="通量值",
    )

    # 设置标题和标签
    ax_z.set_title(f"Z轴切面 (Z = {init_z})")
    ax_z.set_xlabel("X轴")
    ax_z.set_ylabel("Y轴")

    ax_y.set_title(f"Y轴切面 (Y = {init_y})")
    ax_y.set_xlabel("X轴")
    ax_y.set_ylabel("Z轴")

    ax_x.set_title(f"X轴切面 (X = {init_x})")
    ax_x.set_xlabel("Y轴")
    ax_x.set_ylabel("Z轴")

    # 确保坐标轴刻度比例一致
    for ax in [ax_z, ax_y, ax_x]:
        ax.set_xticks(np.linspace(0, max(ax.get_xlim()), 5))
        ax.set_yticks(np.linspace(0, max(ax.get_ylim()), 5))

    # 创建三个滑块的轴位置
    ax_slider_z = plt.axes([0.15, 0.15, 0.7, 0.03])
    ax_slider_y = plt.axes([0.15, 0.1, 0.7, 0.03])
    ax_slider_x = plt.axes([0.15, 0.05, 0.7, 0.03])

    # 创建三个滑块，分别控制X、Y、Z轴切面
    slider_z = Slider(
        ax=ax_slider_z,
        label="Z轴位置",
        valmin=0,
        valmax=dim_z - 1,
        valinit=init_z,
        valstep=1,
    )

    slider_y = Slider(
        ax=ax_slider_y,
        label="Y轴位置",
        valmin=0,
        valmax=dim_y - 1,
        valinit=init_y,
        valstep=1,
    )

    slider_x = Slider(
        ax=ax_slider_x,
        label="X轴位置",
        valmin=0,
        valmax=dim_x - 1,
        valinit=init_x,
        valstep=1,
    )

    # 定义更新函数
    def update_z(val):
        z_pos = int(round(slider_z.val))
        im_z.set_data(flux[:, :, z_pos])
        ax_z.set_title(f"Z轴切面 (Z = {z_pos})")
        fig.canvas.draw_idle()

    def update_y(val):
        y_pos = int(round(slider_y.val))
        im_y.set_data(flux[:, y_pos, :].T)
        ax_y.set_title(f"Y轴切面 (Y = {y_pos})")
        fig.canvas.draw_idle()

    def update_x(val):
        x_pos = int(round(slider_x.val))
        im_x.set_data(flux[x_pos, :, :].T)
        ax_x.set_title(f"X轴切面 (X = {x_pos})")
        fig.canvas.draw_idle()

    # 注册滑块事件
    slider_z.on_changed(update_z)
    slider_y.on_changed(update_y)
    slider_x.on_changed(update_x)

    plt.show()


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


with h5py.File("./1.mat", "r") as mat:
    print("565555", mat.keys())
    flux_injure = mat["CWfluencem_injure_2"]
    flux_injure = np.array(flux_injure)
    flux_liver = mat["CWfluencem_liver"]
    flux_liver = np.array(flux_liver)

    # ===== 关键步骤：选择合适的坐标系变换 =====
    # 请尝试不同的 transform_option，根据投影结果判断哪个是正确的
    # 通常 MATLAB v7.3 数据需要 "option2" 或 "option4"
    TRANSFORM_OPTION = "option4"  # 可改为 option1, option2, option3, option4, option5

    flux_injure_transformed = transform_flux_for_projection(
        flux_injure, TRANSFORM_OPTION
    )
    flux_liver_transformed = transform_flux_for_projection(flux_liver, TRANSFORM_OPTION)

    flux_proj = {}
    liver_flux_proj = {}
    proj_data_path = f"./proj_injure.npz"
    liver_proj_data_path = f"./proj_liver.npz"

    print("\n开始生成投影...")
    import os

    os.makedirs("zhao_result", exist_ok=True)  # 自动创建结果目录
    for angle in [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]:
        print(f"处理角度: {angle}°")
        proj, depth = generate_projection_view_matrix(
            flux_injure_transformed,
            angle,
            200,
            (256, 256),
            (256, 256),
        )
        flux_proj[f"{angle}"] = proj

        # 保存投影图像
        plt.figure(figsize=(8, 6))
        plt.imshow(proj, cmap="hot")
        plt.colorbar(label="强度值")
        plt.title(f"通量投影 - 角度 {angle}°")
        plt.axis("off")
        save_path = f"zhao_result/proj_injure_{angle}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"已保存投影图像: {save_path}")
        plt.close()

        liver_proj, liver_depth = generate_projection_view_matrix(
            flux_liver_transformed,
            angle,
            200,
            (256, 256),
            (256, 256),
        )
        liver_flux_proj[f"{angle}"] = liver_proj

        # 保存肝脏投影图像
        plt.figure(figsize=(8, 6))
        plt.imshow(liver_proj, cmap="hot")
        plt.colorbar(label="强度值")
        plt.title(f"肝脏通量投影 - 角度 {angle}°")
        plt.axis("off")
        save_path = f"zhao_result/proj_liver_{angle}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"已保存肝脏投影图像: {save_path}")
        plt.close()

    np.savez(proj_data_path, **flux_proj)
    np.savez(liver_proj_data_path, **liver_flux_proj)
    print(f"投影数据已保存到 {proj_data_path} 和 {liver_proj_data_path}")
