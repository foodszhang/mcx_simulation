import subprocess
import os
import jdata as jd
import numpy as np
from get_simple_projection import get_projections, get_multi_direction_projections
import concurrent.futures
from gen_mul_projection import generate_projection_view


def gen_other(entry_path, entry, tag_mat):
    result_file = os.path.join(entry_path, f"{entry}.jnii")
    if not os.path.exists(result_file):
        raise Exception(f"{result_file} 未生成！")
    # tag_mat = np.fromfile("../volume_brain.bin")
    # TODO: 这里硬编码了， 改日再改吧
    full_data = jd.loadjd(result_file)
    if len(full_data["NIFTIData"].shape) == 3:
        flux = full_data["NIFTIData"][:, :, :]
    else:
        flux = full_data["NIFTIData"][:, :, :, 0, 0]
    proj_data_path = os.path.join(entry_path, f"proj.npz")
    dep_proj_data_path = os.path.join(entry_path, f"dep_proj.npz")

    # flux_proj = get_multi_direction_projections(flux, tag_mat)

    flux_proj = {}
    depth_proj = {}
    for angle in [-90, -60, -30, 0, 30, 60, 90]:
        proj, depth = generate_projection_view(
            flux,
            angle,
            200,
            (256, 256),
            (256, 256),
        )
        flux_proj[f"{angle}"] = proj
        depth_proj[f"{angle}"] = depth

    np.savez(proj_data_path, **flux_proj)
    np.savez(dep_proj_data_path, **flux_proj)
    return 0


def process_folders(root_dir):
    """
    遍历根目录，进入所有数字命名的子文件夹并执行mcx命令

    参数:
    root_dir: 要遍历的根目录路径
    """
    # 检查根目录是否存在
    if not os.path.exists(root_dir):
        raise Exception(f"错误: 目录 {root_dir} 不存在")

    # 遍历根目录下的所有条目
    #
    tag_mat_path = os.path.join(root_dir, f"volume_brain.bin")
    tag_mat = np.fromfile(tag_mat_path, dtype=np.uint8).reshape([182, 164, 210])
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)
    results = []
    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)

        # 检查是否是目录且名称为纯数字
        if os.path.isdir(entry_path) and entry.isdigit():
            # 构建要执行的命令
            json_file = f"{entry}.json"
            json_path = os.path.join(entry_path, json_file)

            # 检查JSON文件是否存在
            if not os.path.exists(json_path):
                raise Exception(
                    f"警告: {json_file} 在 {entry_path} 中不存在，跳过该文件夹"
                )

            # 执行mcx命令
            try:
                result_file = os.path.join(entry_path, f"{entry}.jnii")
                if not os.path.exists(result_file):
                    raise Exception(f"{result_file} 未生成！")
                # tag_mat = np.fromfile("../volume_brain.bin")
                # TODO: 这里硬编码了， 改日再改吧
                fut = executor.submit(gen_other, entry_path, entry, tag_mat)
                results.append(fut)

            except subprocess.CalledProcessError as e:
                print(f"命令执行失败，错误: {e.stderr}")
            except Exception as e:
                print(f"处理文件夹 {entry_path} 时发生错误: {str(e)}")

    for future in concurrent.futures.as_completed(results):
        try:
            data = future.result()
        except Exception as exc:
            print("%r generated an exception", exec)


if __name__ == "__main__":
    # 替换为你要遍历的根目录路径
    root_directory = "./20251021/"
    process_folders(root_directory)
