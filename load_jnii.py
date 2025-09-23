import jdata as jd

full_data = jd.loadjd("./test.jnii")
flux = full_data["NIFTIData"][:, :, :, 0, 0]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei"]


# 获取矩阵维度

# 获取矩阵维度 (X, Y, Z)，Z是最后一维
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
im_z = ax_z.imshow(flux[:, :, init_z], cmap="viridis", aspect="equal", origin="lower")
im_y = ax_y.imshow(flux[:, init_y, :].T, cmap="viridis", aspect="equal", origin="lower")
im_x = ax_x.imshow(flux[init_x, :, :].T, cmap="viridis", aspect="equal", origin="lower")

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
