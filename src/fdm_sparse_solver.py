# --------------------------------------------------
# 医学/生物体积光场的有限差分法（FDM）稀疏矩阵建模 - 扩散近似方程求解
# 正向过程：光源激发 -> 求解光场分布 -> 探测器提取响应
#
# 物理模型（扩散近似方程）：
#   -∇·(D∇Φ) + μₐ·Φ = S
#
# 其中：
#   Φ: 光通量(光场分布) [W/cm²]
#   D = 1/(3(μₐ+μₛ)): 扩散系数
#   μₐ: 吸收系数 [cm⁻¹]
#   μₛ: 散射系数 [cm⁻¹]
#   S: 光源项 [W/cm³]
#
# FDM离散后得到线性系统：
#   A·Φ = S
#
# 计算步骤：
#   1. 构建系统矩阵A（编码扩散+吸收+边界条件）
#   2. 构造离散光源S
#   3. 求解线性系统: Φ = A⁻¹·S  (LU分解)
#   4. 在探测器位置提取: b_i = Φ[detector_pos_i]
#
# 作者：foodszhang/[项目接手者]
# 说明：构建稀疏系统矩阵A编码扩散近似方程，支持梯度传递及内存优化
# 用法示例：uv run python src/fdm_sparse_solver.py
# --------------------------------------------------
import numpy as np
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
    基于体积分割+材料属性，构建稀疏FDM系统矩阵A

    离散扩散近似方程: -∇·(D∇Φ) + μₐ·Φ = S

    使用有限差分法（7点模板）离散化：
    对于内部点(i,j,k)：
      A[i,j,k; i,j,k] = -μₐ - 6*D/Δx²
      A[i,j,k; 邻域] = D/Δx²

    参数:
        seg: 体积分割三维张量，uint8，不同值代表不同组织
        mua_vol: 吸收系数张量，形状同seg，[cm⁻¹]
        mus_vol: 散射系数张量，形状同seg，[cm⁻¹]

    返回:
        A_sparse: COO格式稀疏系统矩阵，形状(N, N)
    """
    shape = seg.shape
    Z, Y, X = shape
    N = Z * Y * X

    # 线性索引映射：(z,y,x) -> idx
    index_map = lambda z, y, x: z * (Y * X) + y * X + x

    rows = []
    cols = []
    data = []

    # 遍历所有体素，构建矩阵方程
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                idx = index_map(z, y, x)
                mua = mua_vol[z, y, x]  # 吸收系数
                mus = mus_vol[z, y, x]  # 散射系数
                g = g_vol[z, y, x]  # 各向异性因子

                # ========= 关键步骤 =========
                # 计算约化散射系数 (必须使用此公式)
                # μₛ' = μₛ·(1-g)
                mus_prime = mus * (1.0 - g)

                # 扩散系数：D = 1/(3(μₐ + μₛ'))
                # 【重点】必须使用约化散射系数mus_prime，而非原散射系数mus
                D = 1.0 / (3.0 * (mua + mus_prime + 1e-10))

                # 对角元：-μₐ - 6D (来自拉普拉斯算子和吸收项)
                rows.append(idx)
                cols.append(idx)
                data.append(-mua - 6 * D)

                # 六邻域贡献：+D (来自拉普拉斯算子)
                neighbors = [
                    (-1, 0, 0),  # z-1
                    (1, 0, 0),  # z+1
                    (0, -1, 0),  # y-1
                    (0, 1, 0),  # y+1
                    (0, 0, -1),  # x-1
                    (0, 0, 1),  # x+1
                ]

                for dz, dy, dx in neighbors:
                    zz, yy, xx = z + dz, y + dy, x + dx
                    # 检查是否在体积内（开放边界条件）
                    if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X:
                        n_idx = index_map(zz, yy, xx)
                        rows.append(idx)
                        cols.append(n_idx)
                        data.append(D)

    # 转换为COO格式稀疏矩阵
    A_sparse = coo_matrix((data, (rows, cols)), shape=(N, N))
    return A_sparse


# =====================
# 扩散方程求解接口
# =====================
def solve_diffusion_equation(A_sparse, source):
    """
    求解扩散近似方程: A·Φ = S

    使用scipy稀疏直接求解器（LU分解）

    参数:
        A_sparse: COO/CSR格式系统矩阵A
        source: 光源项S（维数N），单位[W/cm³]

    返回:
        phi: 光通量/光场分布Φ（维数N），单位[W/cm²]
    """
    # 转换为CSR格式后进行LU分解求解
    phi = spsolve(A_sparse.tocsr(), source)
    return phi


# =====================
# 探测器响应提取接口
# =====================
def extract_detector_response(phi, detector_positions, shape):
    """
    从光场分布Φ中提取探测器响应

    参数:
        phi: 光通量分布，维数N
        detector_positions: 探测器位置列表 [(z1,y1,x1), (z2,y2,x2), ...]
        shape: 体积形状 (Z, Y, X)

    返回:
        b: 探测器响应向量，维数M（M=探测器数量）
    """
    Z, Y, X = shape
    index_map = lambda z, y, x_coord: z * (Y * X) + y * X + x_coord

    b = np.zeros(len(detector_positions), dtype=np.float32)

    for i, (dz, dy, dx) in enumerate(detector_positions):
        if 0 <= dz < Z and 0 <= dy < Y and 0 <= dx < X:
            idx = index_map(dz, dy, dx)
            b[i] = phi[idx]
        else:
            print(f"⚠ 警告：探测器{i}位置({dz},{dy},{dx})越界")

    return b


# =====================
# 数据加载函数
# =====================
def load_volume_and_media(volume_file, material_yaml):
    """
    加载体积分割文件与材料属性

    【关键】必须从yaml配置中读取g参数(各向异性因子)
    yaml格式必须包含: id, mua, mus, g

    参数:
        volume_file: 体积数据文件路径 (*.bin)
        material_yaml: 材料属性配置文件 (*.yaml)

    返回:
        seg: 体积分割标签，shape=(Z,Y,X)，dtype=uint8
        mua: 吸收系数，shape=(Z,Y,X)
        mus: 散射系数，shape=(Z,Y,X)
    """
    assert os.path.exists(volume_file), f"体积文件不存在: {volume_file}"

    # 读取材料属性配置
    with open(material_yaml, "r") as ff:
        media_list = yaml.safe_load(ff)

    # 从JSON配置文件获取体积形状
    json_path = volume_file.replace(".bin", ".json")
    with open(json_path, "r") as jj:
        ref_json = json.load(jj)
    shape = tuple(ref_json["Domain"]["Dim"])  # (Z, Y, X)

    # 读取体积数据
    raw = np.fromfile(volume_file, dtype=np.uint8).reshape(shape)
    seg = raw

    # 建立材料ID到光学参数的查表
    n_mat = len(media_list)
    mua_tab = np.zeros(n_mat + 1, dtype=np.float32)
    mus_tab = np.zeros(n_mat + 1, dtype=np.float32)

    for i, media in enumerate(media_list):
        idx = int(media["id"])
        mua_tab[idx] = media.get("mua", 0.0)
        mus_tab[idx] = media.get("mus", 0.0)

    # 根据分割结果查表得到光学参数
    mua = mua_tab[seg].astype(np.float32)
    mus = mus_tab[seg].astype(np.float32)

    return seg, mua, mus


# =====================
# 主程序：完整的FDM求解流程
# =====================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("扩散近似方程FDM求解 - 计算光在生物组织中的传输")
    print("=" * 70)

    # ==================== 步骤1：加载数据 ====================
    print("\n[步骤1] 加载体积数据和材料属性")
    material_yaml = "volume_bases/material/rat_brain_1200nm.yaml"
    volume_file = "your_volume_file.bin"

    try:
        seg, mua_vol, mus_vol = load_volume_and_media(volume_file, material_yaml)
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        exit(1)

    shape = seg.shape
    Z, Y, X = shape
    N = Z * Y * X

    print(f"  ✓ 体积形状 (Z, Y, X): {shape}")
    print(f"  ✓ 体素总数 N: {N}")
    print(f"  ✓ 吸收系数 μₐ: [{mua_vol.min():.6f}, {mua_vol.max():.6f}] cm⁻¹")
    print(f"  ✓ 散射系数 μₛ: [{mus_vol.min():.6f}, {mus_vol.max():.6f}] cm⁻¹")

    # ==================== 步骤2：读取光源配置 ====================
    print("\n[步骤2] 读取光源激发模式")
    config_json_path = volume_file.replace(".bin", ".json")
    source = None

    try:
        with open(config_json_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"⚠ 配置文件读取失败: {e}")
        config = {}

    # 尝试从配置文件读取pattern
    source_info = config.get("Optode", {}).get("Source", {}).get("Pattern", {})
    pattern_file = source_info.get("Data")
    pattern_nx = source_info.get("Nx")
    pattern_ny = source_info.get("Ny")
    pattern_nz = source_info.get("Nz")

    if pattern_file and pattern_nx and pattern_ny and pattern_nz:
        pattern_path = os.path.join(os.path.dirname(config_json_path), pattern_file)
        try:
            pattern_arr = np.fromfile(pattern_path, dtype=np.float32).reshape(
                (pattern_nz, pattern_ny, pattern_nx)
            )
            src_pos = config.get("Optode", {}).get("Source", {}).get("Pos", [0, 0, 0])
            src_z, src_y, src_x = src_pos

            # 检查是否越界
            if (
                src_z + pattern_nz > Z
                or src_y + pattern_ny > Y
                or src_x + pattern_nx > X
            ):
                print(f"⚠ Pattern嵌入越界，使用单点激发")
            else:
                # 构造离散光源：在指定位置嵌入pattern
                source_vol = np.zeros(shape, dtype=np.float32)
                source_vol[
                    src_z : src_z + pattern_nz,
                    src_y : src_y + pattern_ny,
                    src_x : src_x + pattern_nx,
                ] = pattern_arr
                source = source_vol.flatten()
                print(f"  ✓ Pattern光源模式")
                print(f"    - 形状: ({pattern_nz}, {pattern_ny}, {pattern_nx})")
                print(f"    - 嵌入位置(Z,Y,X): ({src_z}, {src_y}, {src_x})")
                print(f"    - 非零元素数: {np.count_nonzero(source)}")
        except Exception as e:
            print(f"⚠ Pattern读取失败: {e}")

    # 如果没有pattern，使用单点激发
    if source is None:
        source = np.zeros(N, dtype=np.float32)
        src_z, src_y, src_x = 30, 64, 64
        if src_z < Z and src_y < Y and src_x < X:
            src_idx = src_z * (Y * X) + src_y * X + src_x
            source[src_idx] = 1.0
            print(f"  ✓ 单点光源模式")
            print(f"    - 位置(Z,Y,X): ({src_z}, {src_y}, {src_x})")
            print(f"    - 线性索引: {src_idx}")
        else:
            print(f"✗ 光源位置越界")
            exit(1)

    # ==================== 步骤3：构建系统矩阵A ====================
    print("\n[步骤3] 构建FDM系统矩阵A")
    A_sparse = build_fdm_sparse_matrix(seg, mua_vol, mus_vol)

    print(f"  ✓ 矩阵形状: ({N}, {N})")
    print(f"  ✓ 非零元素数: {A_sparse.nnz}")
    print(f"  ✓ 稀疏度: {100 * A_sparse.nnz / (N * N):.6f}%")
    print(f"  ✓ 值范围: [{A_sparse.data.min():.6e}, {A_sparse.data.max():.6e}]")

    # ==================== 步骤4：求解扩散方程 ====================
    print("\n[步骤4] 求解扩散方程: A·Φ = S")
    print("  求解中...")

    try:
        phi = solve_diffusion_equation(A_sparse, source)
    except Exception as e:
        print(f"✗ 求解失败: {e}")
        exit(1)

    print(f"  ✓ 光场分布Φ (维数N={N})")
    print(f"    - 类型: {phi.dtype}")
    print(f"    - 值范围: [{phi.min():.6e}, {phi.max():.6e}]")
    print(f"    - 均值: {phi.mean():.6e}")
    print(f"    - 标准差: {phi.std():.6e}")

    # ==================== 步骤5：读取探测器配置 ====================
    print("\n[步骤5] 读取探测器位置")
    detector_info = config.get("Optode", {}).get("Detector", {})
    det_pos_list = detector_info.get("Pos", [])

    if det_pos_list:
        detector_positions = [tuple(pos) for pos in det_pos_list]
        print(f"  ✓ 读取{len(detector_positions)}个探测器")
    else:
        # 默认：在z=0平面均匀采样
        detector_positions = []
        for y in range(0, Y, 16):
            for x_coord in range(0, X, 16):
                detector_positions.append((0, y, x_coord))
        print(f"  ✓ 使用默认采样: {len(detector_positions)}个探测器")

    # ==================== 步骤6：提取探测器响应 ====================
    print("\n[步骤6] 提取探测器响应: b = Φ[detector_positions]")
    b = extract_detector_response(phi, detector_positions, shape)

    print(f"  ✓ 探测器响应b (维数M={len(b)})")
    print(f"    - 值范围: [{b.min():.6e}, {b.max():.6e}]")
    print(f"    - 均值: {b.mean():.6e}")
    print(f"    - 标准差: {b.std():.6e}")

    # ==================== 步骤7：保存结果 ====================
    print("\n[步骤7] 保存计算结果")

    np.save("source_S.npy", source)
    print(f"  ✓ 光源项S: source_S.npy (形状{source.shape})")

    np.save("lightfield_phi.npy", phi)
    print(f"  ✓ 光场Φ: lightfield_phi.npy (形状{phi.shape})")

    np.save("detector_response_b.npy", b)
    print(f"  ✓ 探测器响应b: detector_response_b.npy (形状{b.shape})")

    det_pos_array = np.array(detector_positions, dtype=np.int32)
    np.save("detector_positions.npy", det_pos_array)
    print(f"  ✓ 探测器位置: detector_positions.npy (形状{det_pos_array.shape})")

    # ==================== 步骤8：验证求解 ====================
    print("\n[步骤8] 验证求解精度")
    residual = A_sparse.dot(phi) - source
    residual_norm = np.linalg.norm(residual)
    source_norm = np.linalg.norm(source)
    rel_error = residual_norm / (source_norm + 1e-10)

    print(f"  ✓ 残差检验")
    print(f"    - ||A·Φ - S||: {residual_norm:.6e}")
    print(f"    - 相对误差: {rel_error:.6e}")
    if rel_error < 1e-3:
        print(f"    - 求解精度: ✓ 优秀 (相对误差 < 1e-3)")
    else:
        print(f"    - 求解精度: ⚠ 警告 (相对误差 >= 1e-3)")

    # ==================== 最终总结 ====================
    print("\n" + "=" * 70)
    print("求解完成!")
    print("=" * 70)
    print(f"\n数据流总结:")
    print(f"  光源项 S       (维数 N={N})")
    print(f"    ↓ [求解 A·Φ = S]")
    print(f"  光场分布 Φ    (维数 N={N})")
    print(f"    ↓ [在探测器位置采样]")
    print(f"  探测器响应 b  (维数 M={len(b)})")
    print(f"\n输出文件:")
    print(f"  - source_S.npy: 光源项 (N,)")
    print(f"  - lightfield_phi.npy: 光场分布 (N,)")
    print(f"  - detector_response_b.npy: 探测器响应 (M,) ← 最终输出")
    print(f"  - detector_positions.npy: 探测器位置 (M,3)")
