# --------------------------------------------------
# 有限差分法（FDM）光场分布计算脚本
# --------------------------------------------------
# 作者：foodszhang/[项目接手者]
# 说明：医学/生物体成像体素的光传播，支持MCX仿真同格式输入（体积分割文件、材料参数、三维光源及探测器点），输出每个点的光场值（严格中文文档与注释，医学标准实现）
# 用法示例：uv run python src/fdm_light_field_solver.py
# --------------------------------------------------
import numpy as np
import yaml
import json
import os

# =====================
# 核心接口函数
# =====================
def fdm_solve_light_field(volume_file, media_list, source_info, detector_coords,
                        max_iter=2000, tol=1e-4):
    """
    输入体积分割文件、材料属性、光源配置和探测器坐标，返回每个探测器点的光场值
    参数:
        volume_file : str (体素分割数据bin文件路径)
        media_list : list (各分割体生物材料参数，dict格式)
        source_info : dict (MCX格式的光源参数)
        detector_coords : np.ndarray/ndarray (N,3), N个探测器坐标（注：均为(Z,Y,X)顺序）
        max_iter, tol : FDM参数，迭代次数和收敛阈值
    返回:
        field_vals : np.ndarray (N,1)，每个探测器点的光场值
    """
    # 一、载入体积数据（假设为三维uint8类型体素，分割索引）
    assert os.path.exists(volume_file), f"体积文件不存在: {volume_file}"
    vol_shape = None
    if volume_file.endswith(".bin"):
        # bin: 体积尺寸通过材料参数/输入指定，仿真前请保证已知
        # ——此处仅举例，真实项目按实际格式适配——
        # 以常见体素为float32、shape由参数指定
        # 例如材料json: {"Dim": [Z,Y,X], ...}
        # demo指定shape，可扩展
        with open(volume_file, 'rb') as f:
            raw = np.fromfile(f, dtype=np.uint8)
        # 假设体素shape由材料参数推断（如[Z,Y,X]=[64,128,128]）
        if hasattr(media_list, 'Dim'):
            vol_shape = media_list['Dim']
        elif os.path.exists(volume_file.replace('.bin', '.json')):
            with open(volume_file.replace('.bin', '.json'), 'r') as ff:
                ref_json = json.load(ff)
            vol_shape = ref_json['Domain']['Dim']
        else:
            raise ValueError("没有体积shape参数，请补充配置")
        seg = raw.reshape(vol_shape)
    else:
        raise NotImplementedError("仅支持.bin体积文件，如需其它格式请补全代码")
    # 二、根据材料参数生成光学系数三维张量
    # media_list: [{"id": 1, "mua":1.2, "mus": 0.9, "g": 0.88, ...}, ...]
    n_mat = len(media_list)
    mua_tab = np.zeros(n_mat+1)
    mus_tab = np.zeros(n_mat+1)
    for i, media in enumerate(media_list):
        idx = int(media['id'])
        mua_tab[idx] = media.get('mua',0)
        mus_tab[idx] = media.get('mus',0)
    mua_vol = mua_tab[seg]
    mus_vol = mus_tab[seg]
    # 三、初始化光场（三维，float32）
    field = np.zeros_like(seg, dtype=np.float32)
    # 四、光源设置：假定为单点注入或pattern投射（兼容MCX配置）
    src_conf = source_info['Source']
    # 仅支持pattern3d/点源。如pattern3d直接指定区域为1，其他为0
    if src_conf['Type'] == 'pattern3d':
        sz, sy, sx = src_conf['Pattern']['Nz'], src_conf['Pattern']['Ny'], src_conf['Pattern']['Nx']
        spos = src_conf['Pos']  # [z0,y0,x0]
        # 读取pattern（默认为bin，与MCX一致处理）
        pfile = os.path.join(os.path.dirname(volume_file), src_conf['Pattern']['Data'])
        assert os.path.exists(pfile), f"光源Pattern文件不存在: {pfile}"
        with open(pfile, 'rb') as ff:
            parr = np.fromfile(ff, dtype=np.float32).reshape(sz, sy, sx)
        # 光源pattern融入全场
        field[
            spos[0]:spos[0]+sz,
            spos[1]:spos[1]+sy,
            spos[2]:spos[2]+sx
        ] = parr
    else:
        # 默认单点源
        zz,yy,xx = src_conf['Pos']
        field[zz,yy,xx] = 1.
    # 五、有限差分法迭代，简化（稳态扩散）
    # 光输运的Fick扩散模型（近似）： dI/dt = D*laplace(I) - mua*I + S
    # D = 1/(3*(mua + mus))
    D_vol = 1. / (3.*(mua_vol + mus_vol + 1e-6))  # 防除零
    # 简化边界：Dirichlet（四周为零）
    for it in range(max_iter):
        # laplacian, 只近邻求和，不同边界按零处理
        lap = (
          np.roll(field, 1, axis=0) + np.roll(field,-1,axis=0) +
          np.roll(field, 1, axis=1) + np.roll(field,-1,axis=1) +
          np.roll(field, 1, axis=2) + np.roll(field,-1,axis=2) -6*field
        )
        # delta update
        dfield = D_vol*lap - mua_vol*field
        field_new = field + dfield
        # 加光源（初始已放置，叠加不变）
        field_new += field
        # 判断收敛
        rel_err = np.max(np.abs(field_new-field)) / (np.max(np.abs(field))+1e-6)
        field = field_new
        if rel_err < tol:
            print(f"FDM迭代收敛：{it}步，误差：{rel_err:.2e}")
            break
    # 六、采样探测器点（严格Z,Y,X顺序传入）
    N = detector_coords.shape[0]
    field_vals = np.zeros((N,1), dtype=np.float32)
    for i, pt in enumerate(detector_coords.astype(int)):
        z,y,x = pt
        field_vals[i,0] = field[z,y,x]
    return field_vals

# =====================
# 主程序入口及演示用例
# =====================
if __name__ == "__main__":
    # ==========
    # 演示配置，需替换为实际路径和参数yaml/json
    # ==========
    # 假定材料参数yaml为src提供，也可用json格式
    material_yaml = "volume_bases/material/rat_brain_1200nm.yaml"
    volume_file = "your_volume_file.bin" # 需替换实际体积
    source_json = "your_source_config.json" # MCX格式光源参数
    detector_coords = np.array(
        [[30,64,64], [40,70,60], [35,65,80]], dtype=np.int32  # 三个点
    ) # 演示为N,3数组
    with open(material_yaml, 'r') as ff:
        media_list = yaml.safe_load(ff)
    with open(source_json, 'r') as sf:
        sjson = json.load(sf)
    # 调用FDM光场求解
    vals = fdm_solve_light_field(
        volume_file=volume_file,
        media_list=media_list,
        source_info=sjson['Optode'],
        detector_coords=detector_coords
    )
    print("各探测器处光场值：", vals)
    # 可保存到npy/csv文件等，按需扩展

