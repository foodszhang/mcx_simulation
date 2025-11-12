# --------------------------------------------------
# 医学/生物体积光场的有限差分法（FDM）稀疏矩阵建模 Ax=b
# --------------------------------------------------
# 作者：foodszhang/[项目接手者]
# 说明：构建稀疏系统矩阵A，支持梯度传递及内存优化，兼容PyTorch自动微分及分块加速；全中文注释
# 用法示例：uv run python src/fdm_sparse_solver.py
# --------------------------------------------------
import numpy as np
import torch
import os
import yaml
import json
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


# =====================
# 系统矩阵A的稀疏高效构造接口
# =====================
def build_fdm_sparse_matrix(seg, mua_vol, mus_vol):
    """
    基于体积分割+材料属性，构建稀疏FDM系统矩阵A（COO格式，便于Ax=b和深度学习自动微分）
    参数:
        seg: 体积分割三维张量，uint8
        mua_vol, mus_vol: 体素对应吸收/散射系数张量（形状同seg，float）
    返回:
        A_sparse: COO格式稀疏系统矩阵
    """
    shape = seg.shape
    Z, Y, X = shape
    N = Z * Y * X
    # 拉成一维表示，每个点只与自身及六邻域关联
    index_map = lambda z, y, x: z * (Y * X) + y * X + x
    rows = []
    cols = []
    data = []
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                idx = index_map(z, y, x)
                mua = mua_vol[z, y, x]
                mus = mus_vol[z, y, x]
                D = 1.0 / (3.0 * (mua + mus + 1e-6))
                # 对角元
                rows.append(idx)
                cols.append(idx)
                data.append(-mua - 6 * D)
                # 六邻域
                for dz, dy, dx in [
                    (-1, 0, 0),
                    (1, 0, 0),
                    (0, -1, 0),
                    (0, 1, 0),
                    (0, 0, -1),
                    (0, 0, 1),
                ]:
                    zz, yy, xx = z + dz, y + dy, x + dx
                    if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X:
                        n_idx = index_map(zz, yy, xx)
                        rows.append(idx)
                        cols.append(n_idx)
                        data.append(D)
    # COO稀疏矩阵
    A_sparse = coo_matrix((data, (rows, cols)), shape=(N, N))
    return A_sparse


# =====================
# 建议：分块存储A和只处理探测器相关点，极大节省内存
# =====================
def solve_ax_b(A_sparse, b):
    """
    稀疏线性方程Ax=b求解，适合大体积
    参数:
        A_sparse: COO/CSR格式A
        b: 光源/源项
    返回:
        x: 光场分布，一维N
    """
    # 利用scipy高效稀疏解算
    x = spsolve(A_sparse.tocsr(), b)
    return x


# =====================
# 兼容torch自动微分Ax=b模板
# =====================
def torch_sparse_mv(A_sparse, b):
    """
    将scipy COO矩阵转为torch稀疏张量并乘法（便于后续梯度与深度学习流程）
    """
    values = torch.from_numpy(A_sparse.data.astype(np.float32))
    indices = torch.from_numpy(np.vstack((A_sparse.row, A_sparse.col)).astype(np.int64))
    size = A_sparse.shape
    A_torch = torch.sparse_coo_tensor(indices, values, size)
    b_tensor = torch.from_numpy(b.astype(np.float32))
    out = torch.sparse.mm(A_torch, b_tensor.unsqueeze(1))
    return out.squeeze()


# =====================
# 数据加载与演示流程（全中文注释）
# =====================
def load_volume_and_media(volume_file, material_yaml):
    """
    加载体积分割文件与材料属性，返回seg, mua, mus
    """
    assert os.path.exists(volume_file)
    with open(material_yaml, "r") as ff:
        media_list = yaml.safe_load(ff)
    # 采用json侧参数获取体积尺寸
    json_path = volume_file.replace(".bin", ".json")
    with open(json_path, "r") as jj:
        ref_json = json.load(jj)
    shape = ref_json["Domain"]["Dim"]
    raw = np.fromfile(volume_file, dtype=np.uint8).reshape(shape)
    seg = raw
    n_mat = len(media_list)
    mua_tab = np.zeros(n_mat + 1)
    mus_tab = np.zeros(n_mat + 1)
    for i, media in enumerate(media_list):
        idx = int(media["id"])
        mua_tab[idx] = media.get("mua", 0)
        mus_tab[idx] = media.get("mus", 0)
    mua = mua_tab[seg]
    mus = mus_tab[seg]
    return seg, mua, mus


if __name__ == "__main__":
    # 示范流程，全部中文注释
    material_yaml = "volume_bases/material/rat_brain_1200nm.yaml"
    volume_file = "your_volume_file.bin"
    # 光源pattern转为b（以单点激发为例）
    seg, mua_vol, mus_vol = load_volume_and_media(volume_file, material_yaml)
    shape = seg.shape
    N = shape[0] * shape[1] * shape[2]
    b = np.zeros(N, dtype=np.float32)
    # 假定光源在[30,64,64]点注入：
    src_z, src_y, src_x = 30, 64, 64
    src_idx = src_z * (shape[1] * shape[2]) + src_y * shape[2] + src_x
    b[src_idx] = 1.0
    # 生成稀疏A矩阵
    A_sparse = build_fdm_sparse_matrix(seg, mua_vol, mus_vol)
    # 稀疏求解
    x = solve_ax_b(A_sparse, b)
    print(f"光场解算完成，总体场均值: {x.mean():.4e}")
    # 兼容torch自动微分
    torch_field = torch_sparse_mv(A_sparse, b)
    print(f"Torch兼容字段, 总体场均值: {torch_field.mean().item():.4e}")
    # 可依照探测器坐标采样点补充采样流程
    # 全流程严格uv run python执行
