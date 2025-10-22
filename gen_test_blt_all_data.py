import subprocess
import os
import jdata as jd
import numpy as np
from get_simple_projection import get_projections, get_multi_direction_projections


def process_folders(root_dir):
    """
    遍历根目录，进入所有数字命名的子文件夹并执行mcx命令

    参数:
    root_dir: 要遍历的根目录路径
    """
    # 检查根目录是否存在
    if not os.path.exists(root_dir):
        raise Exception(f"错误: 目录 {root_dir} 不存在")
    tag_mat_path = os.path.join(root_dir, f"volume_brain.bin")
    tag_mat = np.fromfile(tag_mat_path, dtype=np.uint8).reshape([182, 164, 210])
    tag_data_path = os.path.join(root_dir, f"volume_brain.npy")
    if not os.path.exists(tag_data_path):
        np.save(tag_data_path, tag_mat)

    # 遍历根目录下的所有条目
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
            if os.path.exists(os.path.join(entry_path, f"{entry}.jnii")):
                continue
            # 执行mcx命令
            try:
                print(f"执行命令: mcx -f {json_file} -a 1 在 {entry_path}")

                # 执行命令并等待完成
                result = subprocess.run(
                    ["mcx", "-f", json_file, "-a", "1"],
                    cwd=entry_path,  # 在子文件夹中执行命令
                    check=True,
                    capture_output=True,
                    text=True,
                )

                # 命令成功执行后要做的事情
                print(f"命令成功完成，输出: {result.stdout}")

                # 这里可以添加命令执行成功后的其他操作
                # 例如: 处理输出文件、记录日志等
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
                flux_data_path = os.path.join(entry_path, f"{entry}_flux.npy")
                proj_data_path = os.path.join(entry_path, f"{entry}_proj.npz")
                # np.save(flux_data_path, flux)
                tag_data_path = os.path.join(root_dir, f"volume_brain.npy")
                if not os.path.exists(tag_data_path):
                    np.save(tag_data_path, tag_mat)

                # flux_proj = get_multi_direction_projections(flux, tag_mat)
                # np.savez(proj_data_path, **flux_proj)

            except subprocess.CalledProcessError as e:
                print(f"命令执行失败，错误: {e.stderr}")
            except Exception as e:
                print(f"处理文件夹 {entry_path} 时发生错误: {str(e)}")


if __name__ == "__main__":
    # 替换为你要遍历的根目录路径
    root_directory = "./20251021/"
    process_folders(root_directory)
