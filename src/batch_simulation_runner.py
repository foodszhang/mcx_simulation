import subprocess
import os
import jdata as jd
import numpy as np


def batch_run_mcx_simulations(root_folder: str, config: dict) -> None:
    """
    批量遍历根目录，自动进入所有数字命名的子目录，批量执行mcx命令并保存仿真结果。

    参数:
        root_folder (str): 根目录路径。
        config (dict): 全局配置参数，统一由主流程入口传入
    """
    import shutil  # 在函数体内部导入避免全局污染

    # ---- MCX命令可用性自动检测与GPU能力判别，兼容WSL/win/linux全版本 ----
    mcx_exec = None
    gpu_supported = False
    # 优先检测mcx, mcx.exe，尝试判断GPU能力
    for name in ["mcx", "mcx.exe"]:
        path = shutil.which(name)
        if path is not None:
            try:
                result = subprocess.run(
                    [name, "-L"], capture_output=True, text=True, check=True
                )
                out = result.stdout.lower() + result.stderr.lower()
                if "gpu" in out or "nvidia" in out:
                    mcx_exec = name
                    gpu_supported = True
                    print(f"已检测到GPU仿真工具：{name}")
                    break
                else:
                    print(f"已检测到{name}，但无GPU能力，将尝试CPU仿真工具（mcxcl）。")
            except Exception as e:
                print(f"检测{name} GPU能力时发生异常：{e}，尝试其他仿真命令。")
    # 若GPU版无效，自动降级为mcxcl/mcxcl.exe（CPU多线程）
    if not gpu_supported:
        for name in ["mcxcl", "mcxcl.exe"]:
            path = shutil.which(name)
            if path is not None:
                mcx_exec = name
                print(f"降级使用CPU仿真工具：{name}")
                break
    if not mcx_exec:
        raise RuntimeError(
            "未检测到任何可用的MCX仿真命令（mcx/mcx.exe/mcxcl/mcxcl.exe），请检查环境变量与工具安装。"
        )

    # ---- 目录与体标签文件准备 ----
    if not os.path.exists(root_folder):
        raise FileNotFoundError(f"错误: 根目录 {root_folder} 不存在")

    volume_bin_name = os.path.basename(
        config.get("generated_bin_path", "volume_true_body.bin")
    )
    volume_bin_path = os.path.join(root_folder, volume_bin_name)
    if not os.path.exists(volume_bin_path):
        raise FileNotFoundError(f"{volume_bin_name} 不存在于 {root_folder}")

    # 预加载体标签矩阵，方便后续复用
    # —— 体标签文件reshape改为自动读取全局config中的volume_shape ——
    volume_shape = config.get("volume_shape", [512, 512, 512])
    assert isinstance(volume_shape, (list, tuple)) and len(volume_shape) == 3, (
        "config['volume_shape']格式错误，需为三元组/list"
    )
    volume_tags = np.fromfile(volume_bin_path, dtype=np.uint8).reshape(volume_shape)

    volume_npy_path = os.path.join(
        root_folder, os.path.splitext(volume_bin_name)[0] + ".npy"
    )
    if not os.path.exists(volume_npy_path):
        np.save(volume_npy_path, volume_tags)

    # ---- 子文件夹批量处理 ----
    def format_filename(template, _id):
        return template.format(id=_id)

    file_naming = config.get(
        "file_naming",
        {
            "normal": {"config": "{id}.json", "result": "{id}.jnii"},
            "noscatter": {"config": "no_{id}.json", "result": "no_{id}.jnii"},
        },
    )

    for sub_name in os.listdir(root_folder):
        sub_folder = os.path.join(root_folder, sub_name)

        # 只处理纯数字命名的目录
        if os.path.isdir(sub_folder) and sub_name.isdigit():
            # 配置文件路径
            json_normal = format_filename(file_naming["normal"]["config"], sub_name)
            path_json_normal = os.path.join(sub_folder, json_normal)

            # 校验标准JSON配置存在，否则跳过
            if not os.path.exists(path_json_normal):
                print(f"警告: 缺少 {json_normal}，跳过 {sub_folder}")
                continue

            result_jnii_path = os.path.join(
                sub_folder, format_filename(file_naming["normal"]["result"], sub_name)
            )
            if os.path.exists(result_jnii_path):
                # 已有仿真结果直接跳过提高性能
                continue

            try:
                # --- 执行标准mcx仿真 ---
                print(f"执行: {mcx_exec} -f {json_normal} -a 1")
                result_normal = subprocess.run(
                    [mcx_exec, "-f", json_normal, "-a", "1"],
                    cwd=sub_folder,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(f"标准仿真输出: {result_normal.stdout}")

                # ---- 仿真输出校验 ----
                if not os.path.exists(result_jnii_path):
                    raise RuntimeError(f"仿真结果未生成: {result_jnii_path}")

                # === 新增无散射仿真（mus=0）处理 ===
                json_no_scatter = format_filename(
                    file_naming["noscatter"]["config"], sub_name
                )
                path_json_no_scatter = os.path.join(sub_folder, json_no_scatter)
                result_no_jnii_path = os.path.join(
                    sub_folder,
                    format_filename(file_naming["noscatter"]["result"], sub_name),
                )
                if os.path.exists(path_json_no_scatter):
                    if os.path.exists(result_no_jnii_path):
                        print(
                            f"已存在无散射仿真结果: {result_no_jnii_path}，跳过无散射仿真。"
                        )
                    else:
                        try:
                            print(f"执行无散射: {mcx_exec} -f {json_no_scatter} -a 1")
                            result_no = subprocess.run(
                                [mcx_exec, "-f", json_no_scatter, "-a", "1"],
                                cwd=sub_folder,
                                check=True,
                                capture_output=True,
                                text=True,
                            )
                            print(f"无散射仿真输出: {result_no.stdout}")
                            if not os.path.exists(result_no_jnii_path):
                                raise RuntimeError(
                                    f"无散射仿真结果未生成: {result_no_jnii_path}"
                                )
                        except subprocess.CalledProcessError as err:
                            print(f"无散射命令执行失败: {err.stderr}")
                        except Exception as ex2:
                            print(f"无散射处理 {sub_folder} 时发生异常: {str(ex2)}")

            except subprocess.CalledProcessError as err:
                print(f"命令执行失败: {err.stderr}")
            except Exception as ex:
                print(f"处理 {sub_folder} 时发生异常: {str(ex)}")


if __name__ == "__main__":
    # 演示主流程，兼容独立脚本执行
    # 临时读取配置文件，便于单文件测试
    import yaml

    # 这里默认读取 config/config.yaml，实际请根据需要变更
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # 替换为你要遍历的根目录路径
    root_directory = "./20251103/"
    batch_run_mcx_simulations(root_directory, config)
