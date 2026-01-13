import subprocess
import os
import jdata as jd
import yaml  # 用于读取 config/config.yaml
import numpy as np
import concurrent.futures
from .gen_mul_projection import generate_projection_view_matrix


# 通用投影函数（标准/无散射，通过sim_type参数控制）
def format_filename(template, _id):
    return template.format(id=_id)


def generate_projection(entry_path, entry, config, sim_type="normal"):
    """
    通用投影生成函数，根据sim_type区分标准仿真/无散射仿真
    参数:
        entry_path: 当前数据子目录路径
        entry: 样本编号/子目录名
        config: 总流程配置（由主流程入口参数层层透传）
        sim_type: 仿真类型，可取"normal"(标准)或"no_scatter"(无散射，mus=0)
    """
    file_naming = config.get(
        "file_naming",
        {
            "normal": {"config": "{id}.json", "result": "{id}.jnii"},
            "noscatter": {"config": "no_{id}.json", "result": "no_{id}.jnii"},
        },
    )
    proj_params = config.get("projection", {})
    # ==== 存储文件名均由 config['projection'] 配置，未设置时自动降级为默认名 ====
    flux_proj_npz = config.get("projection", {}).get("flux_proj_npz", "proj.npz")
    depth_proj_npz = config.get("projection", {}).get("depth_proj_npz", "dep_proj.npz")
    no_flux_proj_npz = config.get("projection", {}).get(
        "no_flux_proj_npz", "no_proj.npz"
    )
    no_depth_proj_npz = config.get("projection", {}).get(
        "no_depth_proj_npz", "no_dep_proj.npz"
    )
    if sim_type == "no_scatter":
        result_file = os.path.join(
            entry_path, format_filename(file_naming["noscatter"]["result"], entry)
        )
        proj_data_path = os.path.join(entry_path, no_flux_proj_npz)
        dep_proj_data_path = os.path.join(entry_path, no_depth_proj_npz)
    else:
        result_file = os.path.join(
            entry_path, format_filename(file_naming["normal"]["result"], entry)
        )
        proj_data_path = os.path.join(entry_path, flux_proj_npz)
        dep_proj_data_path = os.path.join(entry_path, depth_proj_npz)
    if not os.path.exists(result_file):
        raise Exception(f"{result_file} 未生成！")
    full_data = jd.loadjd(result_file)
    if len(full_data["NIFTIData"].shape) == 3:
        flux = full_data["NIFTIData"][:, :, :]
    else:
        flux = full_data["NIFTIData"][:, :, :, 0, 0]
    # 投影参数均从主流程透传的proj_params内读取，彻底去除硬编码
    angles = proj_params.get("angles", [-90, -60, -30, 0, 30, 60, 90])
    print("234234234============", proj_params, angles, flux.shape)
    det_res = tuple(proj_params.get("detector_resolution", (256, 256)))
    flux_proj = {}
    depth_proj = {}
    for angle in angles:
        proj, depth = generate_projection_view_matrix(
            flux,
            angle,
            256,
            det_res,
            det_res,
        )
        flux_proj[f"{angle}"] = proj
        depth_proj[f"{angle}"] = depth
    np.savez(proj_data_path, **flux_proj)
    np.savez(dep_proj_data_path, **depth_proj)
    return 0


def gen_other_all(entry_path, entry, config):
    """
    综合调用通用投影函数，先标准仿真后无散射，配置参数由主流程集中透传。
    参数：
        entry_path: 当前样本子目录路径
        entry: 当前子目录编号
        config: 主流程集中下发的完整配置参数
    """
    # 标准仿真流程
    generate_projection(entry_path, entry, config, sim_type="normal")
    # 无散射仿真流程
    generate_projection(entry_path, entry, config, sim_type="no_scatter")
    return 0


def process_folders(root_dir, config):
    """
    遍历根目录，进入所有数字命名的子文件夹并执行mcx命令，实现批量后处理。所有流程参数均由主流程传入的config字典驱动，避免局部硬编码。
    参数:
    root_dir: 要遍历的根目录路径
    config: 全局流程配置（由main.py集中加载并统一下发）
    """
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
    # tag_mat_path = os.path.join(root_dir, f"volume_brain.bin")
    # tag_mat = np.fromfile(tag_mat_path, dtype=np.uint8).reshape([182, 164, 210])
    # executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)
    results = []
    file_naming = config.get(
        "file_naming",
        {
            "normal": {"config": "{id}.json", "result": "{id}.jnii"},
            "noscatter": {"config": "no_{id}.json", "result": "no_{id}.jnii"},
        },
    )

    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)

        # 检查是否是目录且名称为纯数字
        if os.path.isdir(entry_path) and entry.isdigit():
            json_file = format_filename(file_naming["normal"]["config"], entry)
            json_path = os.path.join(entry_path, json_file)

            # 检查JSON文件是否存在
            if not os.path.exists(json_path):
                raise Exception(
                    f"警告: {json_file} 在 {entry_path} 中不存在，跳过该文件夹"
                )

            try:
                result_file = os.path.join(
                    entry_path, format_filename(file_naming["normal"]["result"], entry)
                )
                if not os.path.exists(result_file):
                    raise Exception(f"{result_file} 未生成！")

                fut = executor.submit(gen_other_all, entry_path, entry, config)
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
    root_directory = "./20251105"
    process_folders(root_directory, config=config)
