# --------------------------------------------------
# 系统矩阵分析工具
# --------------------------------------------------
# 用于检查、分析和验证系统矩阵A的性质
# 支持矩阵统计、条件数计算、稀疏模式可视化等
# --------------------------------------------------

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds, eigs
from pathlib import Path
from typing import Dict, Tuple, Optional
import json


class MatrixAnalyzer:
    """
    系统矩阵分析工具 - 计算矩阵的数学性质和物理特性
    """
    
    def __init__(self, matrix_path: Optional[str] = None, matrix: Optional[sp.csr_matrix] = None):
        """
        初始化矩阵分析器
        :param matrix_path: 矩阵文件路径 (.npz格式)
        :param matrix: 直接提供矩阵对象
        """
        if matrix_path is not None:
            self.matrix = sp.load_npz(matrix_path)
            self.matrix_path = matrix_path
        elif matrix is not None:
            self.matrix = matrix.tocsr()
            self.matrix_path = None
        else:
            raise ValueError("必须提供 matrix_path 或 matrix")
    
    def get_basic_info(self) -> Dict:
        """
        获取矩阵的基本信息
        :return: 包含矩阵维数、非零元素等信息的字典
        """
        info = {
            "shape": tuple(self.matrix.shape),
            "dtype": str(self.matrix.dtype),
            "format": self.matrix.format,
            "nnz": self.matrix.nnz,
            "density": self.matrix.nnz / (self.matrix.shape[0] * self.matrix.shape[1]),
            "memory_bytes": self.matrix.data.nbytes + self.matrix.indices.nbytes + self.matrix.indptr.nbytes,
            "memory_mb": (self.matrix.data.nbytes + self.matrix.indices.nbytes + self.matrix.indptr.nbytes) / (1024**2),
        }
        return info
    
    def get_statistics(self) -> Dict:
        """
        计算矩阵元素的统计量
        :return: 包含min, max, mean, std等的字典
        """
        data = self.matrix.data
        
        stats = {
            "min_value": float(np.min(data)),
            "max_value": float(np.max(data)),
            "mean_value": float(np.mean(data)),
            "std_value": float(np.std(data)),
            "median_value": float(np.median(data)),
            "sum_value": float(np.sum(data)),
            "positive_ratio": float(np.sum(data > 0) / len(data)),
            "negative_ratio": float(np.sum(data < 0) / len(data)),
            "zero_ratio": float(np.sum(data == 0) / len(data)),
        }
        return stats
    
    def get_row_statistics(self) -> Dict:
        """
        计算每行非零元素的统计量
        :return: 包含每行nnz的统计信息
        """
        nnz_per_row = np.diff(self.matrix.indptr)
        
        row_stats = {
            "mean_nnz_per_row": float(np.mean(nnz_per_row)),
            "min_nnz_per_row": int(np.min(nnz_per_row)),
            "max_nnz_per_row": int(np.max(nnz_per_row)),
            "std_nnz_per_row": float(np.std(nnz_per_row)),
        }
        return row_stats
    
    def estimate_condition_number(self, k: int = 6) -> Dict:
        """
        估计矩阵条件数（基于极值奇异值）
        :param k: 计算的奇异值个数
        :return: 包含条件数估计的字典
        """
        try:
            # 计算最大和最小奇异值
            largest = svds(self.matrix, k=1, which='LM', return_singular_vectors=False)
            smallest = svds(self.matrix, k=1, which='SM', return_singular_vectors=False)
            
            cond_est = largest[0] / smallest[0] if smallest[0] > 0 else np.inf
            
            return {
                "condition_number_estimate": float(cond_est),
                "largest_singular_value": float(largest[0]),
                "smallest_singular_value": float(smallest[0]),
            }
        except Exception as e:
            return {"error": str(e)}
    
    def check_symmetry(self) -> Dict:
        """
        检查矩阵是否对称
        :return: 对称性信息字典
        """
        difference = self.matrix - self.matrix.T
        diff_norm = sp.linalg.norm(difference)
        matrix_norm = sp.linalg.norm(self.matrix)
        
        symmetry_check = {
            "is_symmetric": diff_norm < 1e-10 * matrix_norm,
            "symmetry_error": float(diff_norm),
            "relative_error": float(diff_norm / matrix_norm) if matrix_norm > 0 else 0.0,
        }
        return symmetry_check
    
    def check_positive_definiteness(self) -> Dict:
        """
        检查矩阵是否正定
        :return: 正定性检查结果
        """
        # 获取对角线元素
        diag_elements = self.matrix.diagonal()
        
        pd_check = {
            "all_diagonal_positive": bool(np.all(diag_elements > 0)),
            "num_negative_diagonal": int(np.sum(diag_elements < 0)),
            "num_zero_diagonal": int(np.sum(diag_elements == 0)),
            "min_diagonal": float(np.min(diag_elements)),
            "max_diagonal": float(np.max(diag_elements)),
        }
        
        # 尝试进行Cholesky分解以验证正定性
        try:
            from scipy.sparse.linalg import eigsh
            # 计算最小特征值
            lambda_min = eigsh(self.matrix, k=1, which='SM', return_eigenvectors=False)
            pd_check["min_eigenvalue"] = float(lambda_min[0])
            pd_check["likely_positive_definite"] = lambda_min[0] > 1e-10
        except Exception as e:
            pd_check["eigenvalue_check_error"] = str(e)
        
        return pd_check
    
    def get_diagonal_dominance(self) -> Dict:
        """
        检查矩阵的对角线主导性（Diagonal Dominance）
        用于评估迭代求解器的收敛性
        """
        diag = np.abs(self.matrix.diagonal())
        
        # 计算每行的非对角线元素和
        matrix_copy = self.matrix.copy()
        matrix_copy.setdiag(0)
        row_sum = np.abs(matrix_copy).sum(axis=1).A1
        
        # 严格对角线主导：|A[i,i]| > sum(|A[i,j]|, j≠i)
        strictly_dd = np.sum(diag > row_sum)
        
        # 弱对角线主导：|A[i,i]| >= sum(|A[i,j]|, j≠i)
        weakly_dd = np.sum(diag >= row_sum)
        
        dd_check = {
            "strictly_diagonal_dominant": int(strictly_dd),
            "weakly_diagonal_dominant": int(weakly_dd),
            "total_rows": self.matrix.shape[0],
            "ratio_strictly_dd": float(strictly_dd / self.matrix.shape[0]),
            "ratio_weakly_dd": float(weakly_dd / self.matrix.shape[0]),
        }
        
        return dd_check
    
    def get_sparsity_pattern(self) -> Dict:
        """
        分析矩阵的稀疏模式
        :return: 稀疏模式信息
        """
        # 检查带状矩阵特性
        offdiag_dist = []
        for i in range(self.matrix.shape[0]):
            row = self.matrix.getrow(i)
            if row.nnz > 1:
                cols = row.nonzero()[1]
                for col in cols:
                    offdiag_dist.append(abs(i - col))
        
        pattern = {
            "max_bandwidth": int(max(offdiag_dist)) if offdiag_dist else 0,
            "mean_bandwidth": float(np.mean(offdiag_dist)) if offdiag_dist else 0.0,
            "is_banded": len(set(offdiag_dist)) < 20,
        }
        
        return pattern
    
    def print_summary(self, detailed: bool = True):
        """
        打印矩阵分析摘要
        :param detailed: 是否打印详细信息
        """
        print("\n" + "="*70)
        print("系统矩阵分析摘要")
        print("="*70)
        
        # 基本信息
        print("\n【基本信息】")
        basic = self.get_basic_info()
        print(f"  矩阵尺寸: {basic['shape'][0]} × {basic['shape'][1]}")
        print(f"  数据类型: {basic['dtype']}")
        print(f"  非零元素: {basic['nnz']:,}")
        print(f"  稀疏度: {basic['density']*100:.6f}%")
        print(f"  内存占用: {basic['memory_mb']:.2f} MB")
        
        if detailed:
            # 元素统计
            print("\n【元素统计】")
            stats = self.get_statistics()
            print(f"  最小值: {stats['min_value']:.6e}")
            print(f"  最大值: {stats['max_value']:.6e}")
            print(f"  平均值: {stats['mean_value']:.6e}")
            print(f"  标准差: {stats['std_value']:.6e}")
            
            # 行统计
            print("\n【行统计】")
            row_stats = self.get_row_statistics()
            print(f"  每行平均nnz: {row_stats['mean_nnz_per_row']:.2f}")
            print(f"  每行最少nnz: {row_stats['min_nnz_per_row']}")
            print(f"  每行最多nnz: {row_stats['max_nnz_per_row']}")
            
            # 对称性检查
            print("\n【对称性检查】")
            sym = self.check_symmetry()
            print(f"  对称: {'是' if sym['is_symmetric'] else '否'}")
            print(f"  对称误差: {sym['symmetry_error']:.6e}")
            
            # 正定性检查
            print("\n【正定性检查】")
            pd = self.check_positive_definiteness()
            print(f"  对角线全正: {'是' if pd['all_diagonal_positive'] else '否'}")
            if "min_eigenvalue" in pd:
                print(f"  最小特征值: {pd['min_eigenvalue']:.6e}")
                print(f"  可能正定: {'是' if pd.get('likely_positive_definite', False) else '否'}")
            
            # 对角线主导性
            print("\n【对角线主导性】")
            dd = self.get_diagonal_dominance()
            print(f"  严格对角线主导行数: {dd['strictly_diagonal_dominant']}/{dd['total_rows']}")
            print(f"  弱对角线主导行数: {dd['weakly_diagonal_dominant']}/{dd['total_rows']}")
            print(f"  迭代求解器收敛性: {'良好' if dd['ratio_strictly_dd'] > 0.9 else '一般'}")
            
            # 条件数
            print("\n【条件数估计】")
            try:
                cond = self.estimate_condition_number()
                if "error" not in cond:
                    print(f"  条件数(估计): {cond['condition_number_estimate']:.3e}")
                    print(f"  求解数值稳定性: {'稳定' if cond['condition_number_estimate'] < 1e6 else '可能不稳定'}")
            except Exception as e:
                print(f"  计算失败: {e}")
        
        print("\n" + "="*70)
    
    def save_analysis(self, output_path: str):
        """
        保存完整分析结果到JSON文件
        :param output_path: 输出文件路径
        """
        analysis = {
            "basic_info": self.get_basic_info(),
            "statistics": self.get_statistics(),
            "row_statistics": self.get_row_statistics(),
            "symmetry": self.check_symmetry(),
            "positive_definiteness": self.check_positive_definiteness(),
            "diagonal_dominance": self.get_diagonal_dominance(),
            "sparsity_pattern": self.get_sparsity_pattern(),
        }
        
        # 尝试添加条件数信息
        try:
            analysis["condition_number"] = self.estimate_condition_number()
        except:
            pass
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"分析结果已保存: {output_path}")


class VectorAnalyzer:
    """
    向量分析工具 - 分析光源向量和光场解
    """
    
    @staticmethod
    def analyze_vector(vector: np.ndarray, name: str = "Vector") -> Dict:
        """
        分析向量的统计特性
        :param vector: 1D向量
        :param name: 向量名称
        :return: 统计信息字典
        """
        stats = {
            "name": name,
            "length": len(vector),
            "dtype": str(vector.dtype),
            "min": float(np.min(vector)),
            "max": float(np.max(vector)),
            "mean": float(np.mean(vector)),
            "std": float(np.std(vector)),
            "median": float(np.median(vector)),
            "norm_l1": float(np.sum(np.abs(vector))),
            "norm_l2": float(np.linalg.norm(vector)),
            "norm_linf": float(np.max(np.abs(vector))),
            "nonzero_count": int(np.count_nonzero(vector)),
            "sparsity": float(np.count_nonzero(vector) / len(vector)),
        }
        return stats
    
    @staticmethod
    def compare_vectors(v1: np.ndarray, v2: np.ndarray, name1: str = "v1", name2: str = "v2") -> Dict:
        """
        比较两个向量的差异
        :param v1, v2: 两个向量
        :param name1, name2: 向量名称
        :return: 比较结果字典
        """
        if v1.shape != v2.shape:
            raise ValueError("向量维数不匹配")
        
        diff = v1 - v2
        comparison = {
            "name1": name1,
            "name2": name2,
            "difference_l2_norm": float(np.linalg.norm(diff)),
            "difference_relative_l2": float(np.linalg.norm(diff) / np.linalg.norm(v2)) if np.linalg.norm(v2) > 0 else np.inf,
            "max_absolute_difference": float(np.max(np.abs(diff))),
            "mean_absolute_difference": float(np.mean(np.abs(diff))),
            "correlation": float(np.corrcoef(v1, v2)[0, 1]) if len(v1) > 1 else 0.0,
        }
        return comparison
    
    @staticmethod
    def print_vector_analysis(vector: np.ndarray, name: str = "Vector"):
        """
        打印向量分析摘要
        """
        stats = VectorAnalyzer.analyze_vector(vector, name)
        print(f"\n【{name} 分析】")
        print(f"  长度: {stats['length']:,}")
        print(f"  范围: [{stats['min']:.3e}, {stats['max']:.3e}]")
        print(f"  平均: {stats['mean']:.3e}  ±  {stats['std']:.3e}")
        print(f"  L2范数: {stats['norm_l2']:.3e}")
        print(f"  非零元素: {stats['nonzero_count']:,} ({stats['sparsity']*100:.2f}%)")


class ResidualAnalyzer:
    """
    残差分析工具 - 用于验证求解质量
    """
    
    def __init__(self, A: sp.csr_matrix, b: np.ndarray, x: np.ndarray):
        """
        初始化残差分析器
        :param A: 系统矩阵
        :param b: 右端项向量
        :param x: 求解得到的向量
        """
        self.A = A
        self.b = b
        self.x = x
        self.residual = b - A @ x
    
    def get_residual_analysis(self) -> Dict:
        """
        计算残差分析
        :return: 包含残差统计的字典
        """
        b_norm = np.linalg.norm(self.b)
        r_norm = np.linalg.norm(self.residual)
        x_norm = np.linalg.norm(self.x)
        
        analysis = {
            "b_norm": float(b_norm),
            "x_norm": float(x_norm),
            "residual_norm": float(r_norm),
            "relative_residual": float(r_norm / b_norm) if b_norm > 0 else 0.0,
            "residual_min": float(np.min(self.residual)),
            "residual_max": float(np.max(self.residual)),
            "residual_mean": float(np.mean(self.residual)),
            "residual_std": float(np.std(self.residual)),
        }
        return analysis
    
    def print_residual_analysis(self):
        """
        打印残差分析摘要
        """
        analysis = self.get_residual_analysis()
        print("\n【残差分析】")
        print(f"  ||b||: {analysis['b_norm']:.3e}")
        print(f"  ||x||: {analysis['x_norm']:.3e}")
        print(f"  ||r|| = ||b - Ax||: {analysis['residual_norm']:.3e}")
        print(f"  相对残差 ||r||/||b||: {analysis['relative_residual']:.3e}")
        print(f"  求解精度评估: {'优' if analysis['relative_residual'] < 1e-6 else '良' if analysis['relative_residual'] < 1e-4 else '一般'}")


# ============================================================
# 命令行工具
# ============================================================

def main_matrix_analysis(matrix_path: str):
    """
    命令行主函数：分析给定的矩阵文件
    """
    print(f"\n分析矩阵文件: {matrix_path}")
    
    analyzer = MatrixAnalyzer(matrix_path=matrix_path)
    analyzer.print_summary(detailed=True)
    
    # 保存详细分析
    output_json = matrix_path.replace('.npz', '_analysis.json')
    analyzer.save_analysis(output_json)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        matrix_file = sys.argv[1]
        main_matrix_analysis(matrix_file)
    else:
        print("用法: python forward_solver_matrix_tools.py <matrix_file.npz>")
