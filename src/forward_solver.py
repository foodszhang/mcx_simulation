# --------------------------------------------------
# 高散射环境下光传输前向问题求解器
# --------------------------------------------------
# 描述：在高散射环境中使用光传输方程(RTE)求解光场分布
#      系统矩阵A可独立存储为稀疏矩阵格式(CSR/CSC)
# 基于配置：从batch_config_generator中读取source_pattern_in_vol和media信息
# --------------------------------------------------

import numpy as np
import yaml
import os
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.sparse.linalg import spsolve
import json
from typing import Dict, Tuple, Optional


class VolumeReader:
    """
    体素文件读取器 - 从二进制文件中读取和解析三维体积数据
    """
    
    def __init__(self, volume_path: str, volume_shape: Tuple[int, int, int]):
        """
        初始化体素读取器
        :param volume_path: 体素二进制文件路径
        :param volume_shape: 体素数据形状 (Z, Y, X)
        """
        self.volume_path = volume_path
        self.volume_shape = volume_shape
    
    def read_volume(self) -> np.ndarray:
        """
        从二进制文件中读取体素数据
        :return: 体素标签数组，形状为volume_shape
        """
        if not os.path.exists(self.volume_path):
            raise FileNotFoundError(f"体素文件不存在: {self.volume_path}")
        
        volume_data = np.fromfile(self.volume_path, dtype=np.uint8)
        volume_data = volume_data.reshape(self.volume_shape)
        return volume_data


class MediaProperty:
    """
    介质光学属性管理器 - 从YAML文件读取和管理光学参数
    """
    
    def __init__(self, media_yaml_path: str):
        """
        初始化介质属性
        :param media_yaml_path: 材料参数YAML文件路径
        """
        self.media_yaml_path = media_yaml_path
        self.media_list = self._load_media_properties()
    
    def _load_media_properties(self) -> list:
        """
        从YAML文件加载介质光学参数
        :return: 介质参数列表，每项包含mua, mus, g, n等参数
        """
        if not os.path.exists(self.media_yaml_path):
            raise FileNotFoundError(f"材料参数文件不存在: {self.media_yaml_path}")
        
        with open(self.media_yaml_path, 'r', encoding='utf-8') as f:
            media_list = yaml.safe_load(f)
        
        return media_list
    
    def get_property(self, tissue_id: int, prop_name: str) -> float:
        """
        获取指定组织类型的光学参数
        :param tissue_id: 组织编号（对应体素标签）
        :param prop_name: 参数名称 ('mua', 'mus', 'g', 'n')
        :return: 参数值
        """
        if tissue_id >= len(self.media_list):
            raise ValueError(f"组织编号 {tissue_id} 超出范围")
        
        return self.media_list[tissue_id].get(prop_name, 0.0)
    
    def get_scattering_coefficient(self, tissue_id: int) -> float:
        """获取散射系数 mus"""
        return self.get_property(tissue_id, 'mus')
    
    def get_absorption_coefficient(self, tissue_id: int) -> float:
        """获取吸收系数 mua"""
        return self.get_property(tissue_id, 'mua')
    
    def get_anisotropy_factor(self, tissue_id: int) -> float:
        """获取各向异性因子 g"""
        return self.get_property(tissue_id, 'g')
    
    def get_refractive_index(self, tissue_id: int) -> float:
        """获取折射率 n"""
        return self.get_property(tissue_id, 'n')


class LightTransportMatrix:
    """
    光传输系统矩阵生成器 - 构建离散光传输方程的系统矩阵A
    基于RTE(辐射传输方程)的有限体积离散化
    """
    
    def __init__(
        self,
        volume_data: np.ndarray,
        media_property: MediaProperty,
        dx: float = 0.1,
        n_angles: int = 8
    ):
        """
        初始化光传输矩阵生成器
        :param volume_data: 体素标签数组 (Z, Y, X)
        :param media_property: 介质光学属性管理器
        :param dx: 体素尺寸 [mm]
        :param n_angles: 角度离散化数（用于角度方向划分）
        """
        self.volume_data = volume_data
        self.media_property = media_property
        self.dx = dx
        self.n_angles = n_angles
        
        # 体积信息
        self.nz, self.ny, self.nx = volume_data.shape
        self.n_voxels = self.nz * self.ny * self.nx
        
        # 光学参数预计算
        self._precompute_optical_properties()
    
    def _precompute_optical_properties(self):
        """
        预计算每个体素的光学参数以加速矩阵构建
        """
        self.mua = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
        self.mus = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
        self.g_factor = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
        self.n_ref = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
        
        for z in range(self.nz):
            for y in range(self.ny):
                for x in range(self.nx):
                    tissue_id = self.volume_data[z, y, x]
                    self.mua[z, y, x] = self.media_property.get_absorption_coefficient(tissue_id)
                    self.mus[z, y, x] = self.media_property.get_scattering_coefficient(tissue_id)
                    self.g_factor[z, y, x] = self.media_property.get_anisotropy_factor(tissue_id)
                    self.n_ref[z, y, x] = self.media_property.get_refractive_index(tissue_id)
    
    def _linear_index(self, z: int, y: int, x: int) -> int:
        """
        将三维索引转换为线性索引
        :param z, y, x: 三维坐标
        :return: 线性索引
        """
        return z * self.ny * self.nx + y * self.nx + x
    
    def _linear_to_3d(self, idx: int) -> Tuple[int, int, int]:
        """
        将线性索引转换为三维坐标
        :param idx: 线性索引
        :return: (z, y, x) 三维坐标
        """
        z = idx // (self.ny * self.nx)
        remainder = idx % (self.ny * self.nx)
        y = remainder // self.nx
        x = remainder % self.nx
        return z, y, x
    
    def build_system_matrix(self) -> csr_matrix:
        """
        构建系统矩阵A - 基于有限体积法对光传输方程离散化
        
        离散化RTE：
        ∇·(I) + (μa + μs)·I = μs·∫Phase(ω'→ω)·I(ω')·dω' + S
        
        其中：
        - I: 辐亮度 (radiance)
        - μa: 吸收系数
        - μs: 散射系数
        - Phase: 相函数
        - S: 光源项
        
        :return: 稀疏矩阵 A (CSR格式)
        """
        # 计算矩阵维度：每个体素一个方程（漫射近似）
        matrix_size = self.n_voxels
        
        row_indices = []
        col_indices = []
        data_values = []
        
        # 遍历每个体素构建方程
        for idx in range(self.n_voxels):
            z, y, x = self._linear_to_3d(idx)
            
            # 获取当前体素的光学参数
            mu_t = self.mua[z, y, x] + self.mus[z, y, x]  # 总相互作用系数
            
            # 对角元素：表示吸收和散射衰减
            row_indices.append(idx)
            col_indices.append(idx)
            data_values.append(mu_t + 1.0 / self.dx)
            
            # 邻近体素的耦合项（六邻域：±x, ±y, ±z）
            # 实现简化的扩散型耦合（高散射近似）
            neighbors = [
                (z-1, y, x), (z+1, y, x),  # z方向
                (z, y-1, x), (z, y+1, x),  # y方向
                (z, y, x-1), (z, y, x+1),  # x方向
            ]
            
            for nz, ny, nx in neighbors:
                if 0 <= nz < self.nz and 0 <= ny < self.ny and 0 <= nx < self.nx:
                    neighbor_idx = self._linear_index(nz, ny, nx)
                    
                    # 邻近体素的贡献（扩散项）
                    neighbor_coupling = -1.0 / (6.0 * self.dx)
                    
                    row_indices.append(idx)
                    col_indices.append(neighbor_idx)
                    data_values.append(neighbor_coupling)
        
        # 构建CSR格式的稀疏矩阵
        A = csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(matrix_size, matrix_size),
            dtype=np.float32
        )
        
        return A
    
    def build_source_vector(
        self,
        source_pattern: np.ndarray
    ) -> np.ndarray:
        """
        构建右端项向量b（光源项）
        
        :param source_pattern: 光源空间分布 (Z, Y, X)，值范围[0, 1]
        :return: 源向量 b，长度为n_voxels
        """
        b = np.zeros(self.n_voxels, dtype=np.float32)
        
        for z in range(self.nz):
            for y in range(self.ny):
                for x in range(self.nx):
                    idx = self._linear_index(z, y, x)
                    # 光源强度与散射系数成正比（高散射环境）
                    b[idx] = source_pattern[z, y, x] * (1.0 + self.mus[z, y, x])
        
        return b


class ForwardProblemSolver:
    """
    前向问题求解器 - 整合体积读取、介质参数、矩阵构建和求解
    求解高散射环境下的光传输前向问题
    """
    
    def __init__(
        self,
        volume_path: str,
        media_yaml_path: str,
        volume_shape: Tuple[int, int, int],
        dx: float = 0.1
    ):
        """
        初始化前向问题求解器
        :param volume_path: 体素二进制文件路径
        :param media_yaml_path: 材料参数YAML文件路径
        :param volume_shape: 体素形状 (Z, Y, X)
        :param dx: 体素尺寸 [mm]
        """
        self.dx = dx
        
        # 初始化各组件
        self.volume_reader = VolumeReader(volume_path, volume_shape)
        self.media_property = MediaProperty(media_yaml_path)
        
        # 读取体积数据
        self.volume_data = self.volume_reader.read_volume()
        
        # 初始化光传输矩阵生成器
        self.matrix_builder = LightTransportMatrix(
            self.volume_data,
            self.media_property,
            dx=dx
        )
        
        # 系统矩阵（延迟构建）
        self.system_matrix_A = None
        self.source_vector_b = None
        self.solution_phi = None
    
    def build_system(self, source_pattern: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
        """
        构建完整的线性系统 A·φ = b
        
        :param source_pattern: 光源空间分布 (Z, Y, X)
        :return: (系统矩阵A, 源向量b)
        """
        print("构建系统矩阵 A...")
        self.system_matrix_A = self.matrix_builder.build_system_matrix()
        
        print("构建源向量 b...")
        self.source_vector_b = self.matrix_builder.build_source_vector(source_pattern)
        
        print(f"系统矩阵 A: {self.system_matrix_A.shape}, 非零元素: {self.system_matrix_A.nnz}")
        print(f"源向量 b: {self.source_vector_b.shape}, 范数: {np.linalg.norm(self.source_vector_b):.6f}")
        
        return self.system_matrix_A, self.source_vector_b
    
    def solve(self) -> np.ndarray:
        """
        求解线性系统 A·φ = b，得到光场分布φ
        
        :return: 光场解 φ，形状为 (n_voxels,)，需要reshape到(Z, Y, X)
        """
        if self.system_matrix_A is None or self.source_vector_b is None:
            raise RuntimeError("请先调用 build_system() 构建系统")
        
        print("求解线性系统...")
        # 使用稀疏直接求解器
        self.solution_phi = spsolve(
            self.system_matrix_A,
            self.source_vector_b
        ).astype(np.float32)
        
        print(f"光场解: 范数={np.linalg.norm(self.solution_phi):.6f}, "
              f"最大值={np.max(self.solution_phi):.6e}, "
              f"最小值={np.min(self.solution_phi):.6e}")
        
        return self.solution_phi
    
    def get_field_distribution(self) -> np.ndarray:
        """
        获取光场分布，reshape为三维体积
        
        :return: 光场分布 φ，形状为 (Z, Y, X)
        """
        if self.solution_phi is None:
            raise RuntimeError("请先调用 solve() 求解")
        
        nz, ny, nx = self.volume_data.shape
        return self.solution_phi.reshape((nz, ny, nx))
    
    def save_matrix(self, save_path: str):
        """
        将系统矩阵A存储为稀疏矩阵文件（NPZ格式）
        
        :param save_path: 保存文件路径
        """
        if self.system_matrix_A is None:
            raise RuntimeError("请先调用 build_system() 构建系统矩阵")
        
        save_npz(save_path, self.system_matrix_A)
        print(f"系统矩阵已保存: {save_path}")
    
    def load_matrix(self, load_path: str):
        """
        加载之前保存的系统矩阵A
        
        :param load_path: 加载文件路径
        """
        self.system_matrix_A = load_npz(load_path)
        print(f"系统矩阵已加载: {load_path}")
    
    def save_solution(self, save_path: str):
        """
        保存光场分布解
        
        :param save_path: 保存文件路径（支持.npy格式）
        """
        if self.solution_phi is None:
            raise RuntimeError("请先调用 solve() 求解")
        
        field_3d = self.get_field_distribution()
        np.save(save_path, field_3d)
        print(f"光场分布已保存: {save_path}")


class BatchForwardSolver:
    """
    批量前向求解器 - 用于处理batch_config_generator生成的多个配置
    依次求解各配置对应的前向问题
    """
    
    def __init__(
        self,
        config: Dict,
        output_base_dir: str
    ):
        """
        初始化批量求解器
        :param config: 全局配置字典（从batch_config_generator获取）
        :param output_base_dir: 输出基础目录
        """
        self.config = config
        self.output_base_dir = output_base_dir
        
        # 初始化主求解器（共享体积和介质数据）
        self.solver = ForwardProblemSolver(
            volume_path=config["generated_bin_path"],
            media_yaml_path=config["generated_material_yaml_path"],
            volume_shape=tuple(config["volume_shape"]),
            dx=config.get("lengthunit", 0.1)
        )
    
    def solve_batch(self, config_indices: list = None, save_matrices: bool = True):
        """
        批量求解多个配置的前向问题
        
        :param config_indices: 配置索引列表，默认None表示全部
        :param save_matrices: 是否保存系统矩阵
        """
        os.makedirs(self.output_base_dir, exist_ok=True)
        
        for config_idx in config_indices:
            config_subdir = os.path.join(self.output_base_dir, str(config_idx))
            
            # 读取该配置的JSON文件
            json_config_path = os.path.join(
                config_subdir,
                f"{config_idx}.json"
            )
            
            if not os.path.exists(json_config_path):
                print(f"配置文件不存在: {json_config_path}, 跳过")
                continue
            
            print(f"\n{'='*60}")
            print(f"处理配置 {config_idx}: {json_config_path}")
            print('='*60)
            
            # 重建source_pattern_in_vol
            source_pattern = self._reconstruct_source_pattern(config_idx, config_subdir)
            
            # 构建系统
            self.solver.build_system(source_pattern)
            
            # 保存系统矩阵
            if save_matrices:
                matrix_save_path = os.path.join(
                    config_subdir,
                    f"system_matrix_A_{config_idx}.npz"
                )
                self.solver.save_matrix(matrix_save_path)
            
            # 求解
            self.solver.solve()
            
            # 保存光场分布
            field_save_path = os.path.join(
                config_subdir,
                f"light_field_{config_idx}.npy"
            )
            self.solver.save_solution(field_save_path)
            
            # 保存求解信息
            self._save_solve_info(config_idx, config_subdir, source_pattern)
    
    def _reconstruct_source_pattern(
        self,
        config_idx: int,
        config_subdir: str
    ) -> np.ndarray:
        """
        从配置信息重建光源空间分布source_pattern_in_vol
        
        :param config_idx: 配置索引
        :param config_subdir: 配置子目录
        :return: 光源分布数组 (Z, Y, X)
        """
        volume_shape = tuple(self.config["volume_shape"])
        src_range_z = self.config["src_range"]["z"]
        src_range_y = self.config["src_range"]["y"]
        src_range_x = self.config["src_range"]["x"]
        
        # 读取源模式文件
        source_filename = f"source-{config_idx}.bin"
        source_path = os.path.join(config_subdir, source_filename)
        
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"光源文件不存在: {source_path}")
        
        source_arr = np.fromfile(source_path, dtype=np.float32)
        src_voxel_size = (
            self.config["src_range"]["x"][1] - self.config["src_range"]["x"][0],
            self.config["src_range"]["y"][1] - self.config["src_range"]["y"][0],
            self.config["src_range"]["z"][1] - self.config["src_range"]["z"][0],
        )
        source_arr = source_arr.reshape(src_voxel_size)
        source_arr = source_arr.transpose(2, 1, 0)  # 转为ZYX顺序
        
        # 嵌入到体积中
        source_pattern_in_vol = np.zeros(volume_shape, dtype=np.float32)
        source_pattern_in_vol[
            src_range_z[0]:src_range_z[1],
            src_range_y[0]:src_range_y[1],
            src_range_x[0]:src_range_x[1],
        ] = np.where(source_arr > 0.5, 1, 0)
        
        return source_pattern_in_vol
    
    def _save_solve_info(
        self,
        config_idx: int,
        config_subdir: str,
        source_pattern: np.ndarray
    ):
        """
        保存求解信息摘要
        """
        info = {
            "config_idx": config_idx,
            "volume_shape": list(self.config["volume_shape"]),
            "media_yaml": self.config["generated_material_yaml_path"],
            "source_pattern_norm": float(np.linalg.norm(source_pattern)),
            "source_pattern_sum": float(np.sum(source_pattern)),
            "dx": float(self.config.get("lengthunit", 0.1)),
        }
        
        info_path = os.path.join(config_subdir, f"solve_info_{config_idx}.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)


# ============================================================
# 示例使用
# ============================================================

def example_forward_solve():
    """
    示例：单个前向问题求解
    """
    from .load_config import load_config
    
    # 加载全局配置
    config = load_config()
    
    # 初始化求解器
    solver = ForwardProblemSolver(
        volume_path=config["generated_bin_path"],
        media_yaml_path=config["generated_material_yaml_path"],
        volume_shape=tuple(config["volume_shape"]),
        dx=config.get("lengthunit", 0.1)
    )
    
    # 创建示例光源（点光源）
    source_pattern = np.zeros(tuple(config["volume_shape"]), dtype=np.float32)
    src_range_z = config["src_range"]["z"]
    src_range_y = config["src_range"]["y"]
    src_range_x = config["src_range"]["x"]
    
    # 在光源区域中心放置点光源
    z_center = (src_range_z[0] + src_range_z[1]) // 2
    y_center = (src_range_y[0] + src_range_y[1]) // 2
    x_center = (src_range_x[0] + src_range_x[1]) // 2
    source_pattern[z_center, y_center, x_center] = 1.0
    
    # 构建系统
    A, b = solver.build_system(source_pattern)
    
    # 保存系统矩阵
    solver.save_matrix("system_matrix_A.npz")
    
    # 求解
    phi = solver.solve()
    
    # 保存光场分布
    solver.save_solution("light_field_solution.npy")
    
    # 获取三维光场分布
    field_3d = solver.get_field_distribution()
    print(f"光场分布形状: {field_3d.shape}")
    print(f"光场强度范围: [{field_3d.min():.6e}, {field_3d.max():.6e}]")
    
    return solver, field_3d


def example_batch_forward_solve():
    """
    示例：批量前向问题求解
    """
    from .load_config import load_config
    from datetime import datetime
    
    # 加载全局配置
    config = load_config()
    DATE_STRING = datetime.now().strftime("%Y%m%d")
    output_dir = f"./{DATE_STRING}"
    
    # 初始化批量求解器
    batch_solver = BatchForwardSolver(config, output_dir)
    
    # 求解前3个配置
    batch_solver.solve_batch(
        config_indices=[0, 1, 2],
        save_matrices=True
    )


if __name__ == "__main__":
    # 运行示例
    print("启动前向问题求解器示例...")
    example_forward_solve()
