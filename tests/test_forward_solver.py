# --------------------------------------------------
# 前向求解器单元测试和集成测试
# --------------------------------------------------
# 测试框架：pytest
# 运行命令：pytest tests/test_forward_solver.py -v
# --------------------------------------------------

import pytest
import numpy as np
import scipy.sparse as sp
import tempfile
import os
from pathlib import Path

# 导入被测试模块
from src.forward_solver import (
    VolumeReader,
    MediaProperty,
    LightTransportMatrix,
    ForwardProblemSolver,
    BatchForwardSolver
)
from src.forward_solver_matrix_tools import (
    MatrixAnalyzer,
    VectorAnalyzer,
    ResidualAnalyzer
)


class TestVolumeReader:
    """测试体素读取器"""
    
    @pytest.fixture
    def temp_volume_file(self):
        """创建临时体素文件"""
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            # 创建随机体素数据
            volume_data = np.random.randint(0, 4, size=(10, 12, 15), dtype=np.uint8)
            volume_data.tofile(f.name)
            yield f.name, (10, 12, 15), volume_data
        os.unlink(f.name)
    
    def test_read_volume(self, temp_volume_file):
        """测试读取体素文件"""
        file_path, shape, original_data = temp_volume_file
        
        reader = VolumeReader(file_path, shape)
        loaded_data = reader.read_volume()
        
        assert loaded_data.shape == shape
        assert loaded_data.dtype == np.uint8
        assert np.array_equal(loaded_data, original_data)
    
    def test_file_not_found(self):
        """测试文件不存在时的错误处理"""
        with pytest.raises(FileNotFoundError):
            reader = VolumeReader("nonexistent.bin", (10, 10, 10))
            reader.read_volume()


class TestMediaProperty:
    """测试介质属性管理器"""
    
    @pytest.fixture
    def temp_media_file(self):
        """创建临时介质参数文件"""
        import yaml
        
        media_list = [
            {"mua": 0.0, "mus": 0.0, "g": 1.0, "n": 1.0},
            {"mua": 0.0921, "mus": 0.619, "g": 0.9, "n": 1.37},
            {"mua": 0.02083, "mus": 2.49, "g": 0.9, "n": 1.37},
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as f:
            yaml.dump(media_list, f)
            yield f.name, media_list
        
        os.unlink(f.name)
    
    def test_load_media_properties(self, temp_media_file):
        """测试加载介质参数"""
        file_path, expected_data = temp_media_file
        
        media = MediaProperty(file_path)
        assert len(media.media_list) == len(expected_data)
    
    def test_get_property(self, temp_media_file):
        """测试获取单个属性"""
        file_path, expected_data = temp_media_file
        
        media = MediaProperty(file_path)
        mua = media.get_absorption_coefficient(1)
        mus = media.get_scattering_coefficient(1)
        
        assert abs(mua - expected_data[1]["mua"]) < 1e-6
        assert abs(mus - expected_data[1]["mus"]) < 1e-6
    
    def test_invalid_tissue_id(self, temp_media_file):
        """测试无效的组织ID"""
        file_path, _ = temp_media_file
        
        media = MediaProperty(file_path)
        with pytest.raises(ValueError):
            media.get_absorption_coefficient(999)


class TestLightTransportMatrix:
    """测试光传输矩阵生成器"""
    
    @pytest.fixture
    def setup_matrix_builder(self):
        """创建矩阵构建器"""
        # 创建小规模测试数据
        volume_data = np.zeros((5, 5, 5), dtype=np.uint8)
        volume_data[1:4, 1:4, 1:4] = 1  # 在中心放置组织
        
        # 创建简单的介质参数
        import yaml
        media_list = [
            {"mua": 0.01, "mus": 0.1, "g": 0.9, "n": 1.0},
            {"mua": 0.05, "mus": 0.5, "g": 0.9, "n": 1.37},
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as f:
            yaml.dump(media_list, f)
            media_file = f.name
        
        try:
            media = MediaProperty(media_file)
            builder = LightTransportMatrix(volume_data, media, dx=0.1)
            yield builder
        finally:
            os.unlink(media_file)
    
    def test_matrix_construction(self, setup_matrix_builder):
        """测试系统矩阵构建"""
        builder = setup_matrix_builder
        A = builder.build_system_matrix()
        
        # 检查矩阵大小
        assert A.shape == (125, 125)  # 5^3 = 125
        
        # 检查稀疏性
        assert A.nnz < 125 * 125  # 应该是稀疏的
        
        # 检查格式
        assert A.format == 'csr'
    
    def test_source_vector(self, setup_matrix_builder):
        """测试源向量构建"""
        builder = setup_matrix_builder
        
        source = np.zeros((5, 5, 5), dtype=np.float32)
        source[2, 2, 2] = 1.0
        
        b = builder.build_source_vector(source)
        
        # 检查向量大小
        assert len(b) == 125
        
        # 检查非零元素
        assert np.count_nonzero(b) > 0
    
    def test_optical_properties_precomputation(self, setup_matrix_builder):
        """测试光学参数的预计算"""
        builder = setup_matrix_builder
        
        assert builder.mua.shape == (5, 5, 5)
        assert builder.mus.shape == (5, 5, 5)
        assert np.all(builder.mua >= 0)
        assert np.all(builder.mus >= 0)


class TestMatrixAnalyzer:
    """测试矩阵分析工具"""
    
    @pytest.fixture
    def sample_sparse_matrix(self):
        """创建示例稀疏矩阵"""
        # 创建对称的稀疏矩阵
        size = 100
        diag = np.ones(size) * 10
        offdiag = np.ones(size - 1) * (-1)
        
        A = sp.diags([diag, offdiag, offdiag], [0, 1, -1], shape=(size, size), format='csr')
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            sp.save_npz(f.name, A)
            yield f.name, A
        
        os.unlink(f.name)
    
    def test_basic_info(self, sample_sparse_matrix):
        """测试基本信息提取"""
        file_path, original_matrix = sample_sparse_matrix
        
        analyzer = MatrixAnalyzer(matrix_path=file_path)
        info = analyzer.get_basic_info()
        
        assert info['shape'] == original_matrix.shape
        assert info['nnz'] == original_matrix.nnz
        assert info['density'] > 0
        assert info['memory_mb'] > 0
    
    def test_statistics(self, sample_sparse_matrix):
        """测试统计信息"""
        file_path, _ = sample_sparse_matrix
        
        analyzer = MatrixAnalyzer(matrix_path=file_path)
        stats = analyzer.get_statistics()
        
        assert 'min_value' in stats
        assert 'max_value' in stats
        assert 'mean_value' in stats
        assert stats['max_value'] >= stats['min_value']
    
    def test_symmetry_check(self, sample_sparse_matrix):
        """测试对称性检查"""
        file_path, _ = sample_sparse_matrix
        
        analyzer = MatrixAnalyzer(matrix_path=file_path)
        sym_check = analyzer.check_symmetry()
        
        assert 'is_symmetric' in sym_check
        assert 'symmetry_error' in sym_check
    
    def test_diagonal_dominance(self, sample_sparse_matrix):
        """测试对角线主导性"""
        file_path, _ = sample_sparse_matrix
        
        analyzer = MatrixAnalyzer(matrix_path=file_path)
        dd = analyzer.get_diagonal_dominance()
        
        assert 'strictly_diagonal_dominant' in dd
        assert dd['total_rows'] == 100


class TestVectorAnalyzer:
    """测试向量分析工具"""
    
    def test_vector_analysis(self):
        """测试向量分析"""
        vector = np.random.randn(1000)
        stats = VectorAnalyzer.analyze_vector(vector, "test_vector")
        
        assert stats['length'] == 1000
        assert 'min' in stats
        assert 'max' in stats
        assert 'norm_l2' in stats
    
    def test_vector_comparison(self):
        """测试向量比较"""
        v1 = np.array([1, 2, 3, 4, 5], dtype=float)
        v2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1], dtype=float)
        
        comparison = VectorAnalyzer.compare_vectors(v1, v2, "v1", "v2")
        
        assert 'difference_l2_norm' in comparison
        assert 'correlation' in comparison
        assert comparison['difference_l2_norm'] > 0


class TestResidualAnalyzer:
    """测试残差分析工具"""
    
    def test_residual_analysis(self):
        """测试残差分析"""
        # 创建一个简单的线性系统
        A = sp.diags([10, -1, -1], [0, 1, -1], shape=(10, 10), format='csr')
        b = np.ones(10)
        x = np.ones(10) * 0.1  # 近似解
        
        analyzer = ResidualAnalyzer(A, b, x)
        analysis = analyzer.get_residual_analysis()
        
        assert 'residual_norm' in analysis
        assert 'relative_residual' in analysis
        assert analysis['residual_norm'] >= 0


class TestIntegration:
    """集成测试"""
    
    @pytest.fixture
    def temp_simulation_setup(self):
        """创建临时仿真环境"""
        import yaml
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        # 创建体素文件
        volume_shape = (10, 12, 15)
        volume_data = np.random.randint(0, 2, size=volume_shape, dtype=np.uint8)
        volume_file = os.path.join(temp_dir, "volume.bin")
        volume_data.tofile(volume_file)
        
        # 创建介质参数文件
        media_list = [
            {"mua": 0.01, "mus": 0.1, "g": 0.9, "n": 1.0},
            {"mua": 0.05, "mus": 0.5, "g": 0.9, "n": 1.37},
        ]
        media_file = os.path.join(temp_dir, "media.yaml")
        with open(media_file, 'w') as f:
            yaml.dump(media_list, f)
        
        yield temp_dir, volume_file, media_file, volume_shape
        
        # 清理
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_forward_solver_basic(self, temp_simulation_setup):
        """测试前向求解器的基本功能"""
        temp_dir, volume_file, media_file, volume_shape = temp_simulation_setup
        
        # 创建求解器
        solver = ForwardProblemSolver(
            volume_path=volume_file,
            media_yaml_path=media_file,
            volume_shape=volume_shape,
            dx=0.1
        )
        
        # 创建光源
        source = np.zeros(volume_shape, dtype=np.float32)
        source[5, 6, 7] = 1.0
        
        # 构建系统
        A, b = solver.build_system(source)
        assert A.shape[0] == np.prod(volume_shape)
        assert len(b) == np.prod(volume_shape)
        
        # 求解
        phi = solver.solve()
        assert len(phi) == np.prod(volume_shape)
        
        # 获取结果
        field_3d = solver.get_field_distribution()
        assert field_3d.shape == volume_shape
    
    def test_matrix_storage_and_loading(self, temp_simulation_setup):
        """测试矩阵的保存和加载"""
        temp_dir, volume_file, media_file, volume_shape = temp_simulation_setup
        
        # 创建求解器并构建系统
        solver = ForwardProblemSolver(
            volume_path=volume_file,
            media_yaml_path=media_file,
            volume_shape=volume_shape,
            dx=0.1
        )
        
        source = np.zeros(volume_shape, dtype=np.float32)
        source[5, 6, 7] = 1.0
        
        A_original, b = solver.build_system(source)
        
        # 保存矩阵
        matrix_file = os.path.join(temp_dir, "matrix.npz")
        solver.save_matrix(matrix_file)
        assert os.path.exists(matrix_file)
        
        # 加载矩阵
        A_loaded = sp.load_npz(matrix_file)
        
        # 验证矩阵相同
        diff = A_original - A_loaded
        assert sp.linalg.norm(diff) < 1e-10


# ============================================================
# 性能测试
# ============================================================

@pytest.mark.performance
class TestPerformance:
    """性能测试"""
    
    def test_matrix_construction_speed(self):
        """测试矩阵构建速度"""
        import yaml
        import time
        
        # 创建中等规模问题
        volume_shape = (30, 30, 30)
        volume_data = np.zeros(volume_shape, dtype=np.uint8)
        
        media_list = [
            {"mua": 0.01, "mus": 0.1, "g": 0.9, "n": 1.0},
            {"mua": 0.05, "mus": 0.5, "g": 0.9, "n": 1.37},
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as f:
            yaml.dump(media_list, f)
            media_file = f.name
        
        try:
            media = MediaProperty(media_file)
            
            # 测试构建时间
            start_time = time.time()
            builder = LightTransportMatrix(volume_data, media, dx=0.1)
            A = builder.build_system_matrix()
            elapsed = time.time() - start_time
            
            # 断言性能要求
            assert elapsed < 10, f"矩阵构建耗时过长: {elapsed:.2f}秒"
            print(f"矩阵构建耗时: {elapsed:.2f}秒")
        
        finally:
            os.unlink(media_file)


# ============================================================
# 运行所有测试
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
