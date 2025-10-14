import os
import subprocess


def find_jnii_files_with_size_one(root_dir):
    """
    遍历指定目录及其子目录，查找所有后缀为.jnii且大小为1字节的文件

    参数:
        root_dir: 要开始搜索的根目录

    返回:
        符合条件的文件路径列表
    """
    result = []

    # 检查目录是否存在
    if not os.path.exists(root_dir):
        print(f"错误: 目录 '{root_dir}' 不存在")
        return result

    if not os.path.isdir(root_dir):
        print(f"错误: '{root_dir}' 不是一个目录")
        return result

    # 遍历目录
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查文件后缀是否为.jnii
            if filename.lower().endswith(".jnii"):
                file_path = os.path.join(dirpath, filename)
                try:
                    # 获取文件大小（字节）
                    file_size = os.path.getsize(file_path)
                    if file_size == 1:
                        result.append(filename)
                except OSError as e:
                    print(f"无法访问文件 '{file_path}': {e}")

    return result


if __name__ == "__main__":
    import sys

    # 确定要搜索的目录，如果未提供则使用当前目录
    if len(sys.argv) > 1:
        search_dir = sys.argv[1]
    else:
        search_dir = os.getcwd()
        print(f"未指定目录，将搜索当前目录: {search_dir}")

    # 查找符合条件的文件
    jnii_files = find_jnii_files_with_size_one(search_dir)

    # 输出结果
    if jnii_files:
        print(f"\n找到 {len(jnii_files)} 个大小为1字节的.jnii文件:")
        for file_path in jnii_files:
            print(f"- {file_path}")
            result = subprocess.run(
                ["mcx", "-f", f"{file_path.split('.')[0]}.json", "-a", "1"],
                cwd=search_dir,  # 在子文件夹中执行命令
                check=True,
                capture_output=True,
                text=True,
            )

    else:
        print("\n未找到大小为1字节的.jnii文件")
