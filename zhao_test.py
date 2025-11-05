## <uv inject="ephemeral">
# PEP 723 (dependencies declaration)
__version__ = "1.0.0"
__dependencies__ = [
    "numpy >= 1.20.0",
    "h5py >= 3.0",
    "matplotlib >= 3.1.0",
    "numba >= 0.55.0",
    "scipy >= 1.5.0",
]
# End inject.

import numpy as np
import h5py
import os
import re
from multiprocessing import Pool
from gen_mul_projection import generate_projection_view_matrix
import matplotlib.pyplot as plt
import scipy.io as sio  # 用于保存投影结果到MATLAB .mat文件


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
    # print(f"原始flux形状: {flux.shape}")

    if transform_option == "option1":
        # 直接使用，无变换
        transformed = flux.copy()
        # print("使用 option1: 直接使用原始数据（无变换）")

    elif transform_option == "option2":
        # 反转所有轴顺序（MATLAB v7.3常需要此操作）
        transformed = np.transpose(flux, (2, 1, 0))
        # print("使用 option2: transpose((2,1,0)) - 反转轴顺序")

    elif transform_option == "option3":
        # 交换X和Y轴
        transformed = np.transpose(flux, (1, 0, 2))
        # print("使用 option3: transpose((1,0,2)) - 交换X和Y轴")

    elif transform_option == "option4":
        # MCX标准格式调整
        # print("使用 option4: transpose((2,0,1)) - MCX标准调整")
        transformed = np.transpose(flux, (2, 0, 1))

    elif transform_option == "option5":
        # 带翻转的变换：先transpose后flip Z轴
        transformed = np.transpose(flux, (2, 1, 0))
        transformed = np.flip(transformed, axis=2)
        # print("使用 option5: transpose((2,1,0)) + flip Z轴")

    else:
        raise ValueError(f"未知的变换选项: {transform_option}")

    # print(f"变换后flux形状: {transformed.shape}")
    return transformed


def process_mat_file(
    file_path, output_dir, save_format="npz", camera_distance=200, resolution=(256, 256)
):
    # ... existing logic ...
    """
    处理单个 MATLAB 文件，批量产生并保存 (T, A, H, W) 的投影。（支持npz+mat+png）
    """
    with h5py.File(file_path, "r") as mat:
        flux_injure = np.array(mat["source_time_injure_2"])
        flux_liver = np.array(mat["source_time_liver"])
    print(f"开始处理：{file_path}")

    transform_option = "option4"
    angles = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]

    n_time = flux_injure.shape[0]
    n_angle = len(angles)
    H, W = resolution

    # 预分配输出数组 (n_time, n_angle, H, W)
    injure_proj_arr = np.zeros((n_time, n_angle, H, W), dtype=np.float32)
    liver_proj_arr = np.zeros((n_time, n_angle, H, W), dtype=np.float32)

    for t in range(n_time):
        injure_trans = transform_flux_for_projection(flux_injure[t], transform_option)
        liver_trans = transform_flux_for_projection(flux_liver[t], transform_option)
        for ai, angle in enumerate(angles):
            inj_proj, _ = generate_projection_view_matrix(
                injure_trans, angle, camera_distance, resolution, resolution
            )
            liv_proj, _ = generate_projection_view_matrix(
                liver_trans, angle, camera_distance, resolution, resolution
            )
            injure_proj_arr[t, ai] = inj_proj
            liver_proj_arr[t, ai] = liv_proj

    # 输出主文件名
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # 保存npz
    if save_format == "npz":
        np.savez_compressed(
            os.path.join(output_dir, f"{base_name}_injure_projection.npz"),
            projection=injure_proj_arr,
            angles=np.array(angles),
            time=np.arange(n_time),
        )
        np.savez_compressed(
            os.path.join(output_dir, f"{base_name}_liver_projection.npz"),
            projection=liver_proj_arr,
            angles=np.array(angles),
            time=np.arange(n_time),
        )

    # 保存mat
    if save_format == "mat":
        sio.savemat(
            os.path.join(output_dir, f"{base_name}_injure_projection.mat"),
            {
                "projection": injure_proj_arr,
                "angles": angles,
                "time": np.arange(n_time),
            },
        )
        sio.savemat(
            os.path.join(output_dir, f"{base_name}_liver_projection.mat"),
            {"projection": liver_proj_arr, "angles": angles, "time": np.arange(n_time)},
        )

    # png输出组织，主文件夹(injure/liver)/每个时间点t/每个角度一张图
    if save_format == "png":
        out_injure_dir = os.path.join(output_dir, f"{base_name}_injure_png")
        out_liver_dir = os.path.join(output_dir, f"{base_name}_liver_png")
        for t in range(n_time):
            t_injure_dir = os.path.join(out_injure_dir, f"t{t}")
            t_liver_dir = os.path.join(out_liver_dir, f"t{t}")
            os.makedirs(t_injure_dir, exist_ok=True)
            os.makedirs(t_liver_dir, exist_ok=True)
            for ai, angle in enumerate(angles):
                inj_png = injure_proj_arr[t, ai]
                liv_png = liver_proj_arr[t, ai]
                inj_png_path = os.path.join(t_injure_dir, f"proj_angle_{angle}.png")
                liv_png_path = os.path.join(t_liver_dir, f"proj_angle_{angle}.png")
                plt.imsave(inj_png_path, inj_png, cmap="hot")
                plt.imsave(liv_png_path, liv_png, cmap="hot")
                print(f"保存 {inj_png_path}")
                print(f"保存 {liv_png_path}")
    print(f"处理完毕：{file_path}")


def main(
    input_dir="./data",
    output_dir="./output",
    save_format="npz",
    processes=4,
    camera_distance=200,
    resolution=(256, 256),
):
    """
    批量处理一个目录中的多个 MATLAB 文件。

    参数:
        input_dir (str): 要搜索 MATLAB 文件的根目录，默认为'./data'。
        output_dir (str): 保存处理后文件的目录，默认为'./output'。
        save_format (str): 输出格式，默认为'npz'。
        processes (int): 使用的进程数量，默认为4。
        camera_distance (int): 相机与物体的距离。
        resolution (tuple): 投影分辨率。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files_to_process = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if re.match(r"\d+\.mat$", f):
                files_to_process.append(os.path.join(root, f))

    with Pool(processes=processes) as pool:
        pool.starmap(
            process_mat_file,
            [
                (f, output_dir, save_format, camera_distance, resolution)
                for f in files_to_process
            ],
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量处理 MATLAB 文件以生成投影。")
    parser.add_argument(
        "--input_dir", type=str, default="./data", help="包含 .mat 文件的目录。"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="保存输出的目录。"
    )
    parser.add_argument(
        "--save_format",
        type=str,
        choices=["npz", "mat", "png"],
        default="npz",
        help="输出格式(npz/mat/png)。",
    )
    parser.add_argument("--processes", type=int, default=4, help="使用的进程数量。")
    parser.add_argument(
        "--camera_distance", type=int, default=200, help="相机与物体的距离。"
    )
    parser.add_argument(
        "--resolution", type=tuple, default=(256, 256), help="投影分辨率。"
    )

    args = parser.parse_args()
    main(
        args.input_dir,
        args.output_dir,
        args.save_format,
        args.processes,
        args.camera_distance,
        args.resolution,
    )
