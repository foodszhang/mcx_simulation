import subprocess
import os
import jdata as jd
import numpy as np


def batch_run_mcx_simulations(root_folder: str) -> None:
    """
    批量遍历根目录，自动进入所有数字命名的子目录，批量执行mcx命令并保存仿真结果。

    参数:
        root_folder (str): 根目录路径。
    """
    # ---- 目录与体标签文件准备 ----
    if not os.path.exists(root_folder):
        raise FileNotFoundError(f"错误: 根目录 {root_folder} 不存在")
    volume_bin_path = os.path.join(root_folder, "volume_brain.bin")
    if not os.path.exists(volume_bin_path):
        raise FileNotFoundError(f"volume_brain.bin 不存在于 {root_folder}")

    # 预加载体标签矩阵，方便后续复用
    volume_tags = np.fromfile(volume_bin_path, dtype=np.uint8).reshape([182, 164, 210])
    volume_npy_path = os.path.join(root_folder, "volume_brain.npy")
    if not os.path.exists(volume_npy_path):
        np.save(volume_npy_path, volume_tags)

    # ---- 子文件夹批量处理 ----
    for sub_name in os.listdir(root_folder):
        sub_folder = os.path.join(root_folder, sub_name)

        # 只处理纯数字命名的目录
        if os.path.isdir(sub_folder) and sub_name.isdigit():
            # 配置文件路径
            json_normal = f"{sub_name}.json"
            path_json_normal = os.path.join(sub_folder, json_normal)

            # 校验标准JSON配置存在，否则跳过
            if not os.path.exists(path_json_normal):
                print(f"警告: 缺少 {json_normal}，跳过 {sub_folder}")
                continue

            result_jnii_path = os.path.join(sub_folder, f"{sub_name}.jnii")
            if os.path.exists(result_jnii_path):
                # 已有仿真结果直接跳过提高性能
                continue

            try:
                # --- 执行标准mcx仿真 ---
                print(f"执行: mcx -f {json_normal} -a 1 @ {sub_folder}")
                result_normal = subprocess.run(
                    ["mcx", "-f", json_normal, "-a", "1"],
                    cwd=sub_folder,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(f"标准仿真输出: {result_normal.stdout}")

                # ---- 仿真输出校验 ----
                if not os.path.exists(result_jnii_path):
                    raise RuntimeError(f"仿真结果未生成: {result_jnii_path}")

                # ---- 结果数据加载与处理 ----
                sim_data = jd.loadjd(result_jnii_path)
                flux_map = (
                    sim_data["NIFTIData"]
                    if len(sim_data["NIFTIData"].shape) == 3
                    else sim_data["NIFTIData"][:, :, :, 0, 0]
                )

                # 进一步的投影与后处理（根据具体需求修改）
                flux_npy_path = os.path.join(sub_folder, f"{sub_name}_flux.npy")
                np.save(flux_npy_path, flux_map)

            except subprocess.CalledProcessError as err:
                print(f"命令执行失败: {err.stderr}")
            except Exception as ex:
                print(f"处理 {sub_folder} 时发生异常: {str(ex)}")


if __name__ == "__main__":
    # 替换为你要遍历的根目录路径
    root_directory = "./20251103/"
    batch_run_mcx_simulations(root_directory)
