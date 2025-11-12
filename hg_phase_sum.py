import numpy as np
from scipy.integrate import quad
import math


def hg_phase_function(mu, g):
    """
    定义HG相位函数（概率密度函数）
    参数：
        mu: cos(theta)，散射角的余弦值
        g: 不对称因子（-1 ≤ g ≤ 1）
    返回：
        p(mu): HG相位函数在mu处的概率密度
    """
    numerator = 1 - g**2
    denominator = 2 * (1 + g**2 - 2 * g * mu) ** (3 / 2)
    return numerator / denominator


def calculate_scattering_probability(g, theta_min_deg, theta_max_deg):
    """
    计算指定散射角范围内的散射概率（数值积分）
    参数：
        g: 不对称因子
        theta_min_deg: 最小散射角（度）
        theta_max_deg: 最大散射角（度）
    返回：
        prob: 散射概率（0~1）
    """
    # 将角度转换为弧度，再计算对应的mu（cos(theta)）
    theta_min = math.radians(theta_min_deg)
    theta_max = math.radians(theta_max_deg)
    mu_min = math.cos(theta_max)  # 注意：theta越大，mu越小（cos函数单调性）
    mu_max = math.cos(theta_min)

    # 对HG相位函数在[mu_min, mu_max]区间积分（概率=密度函数的积分）
    prob, error = quad(
        hg_phase_function,  # 被积函数
        mu_min,
        mu_max,  # 积分区间（mu从mu_min到mu_max）
        args=(g,),  # 传递给被积函数的额外参数（g）
    )

    return prob, error


# --------------------------
# 示例：计算g=0.9时，0~1°的散射概率
# --------------------------
if __name__ == "__main__":
    g = 0.9
    theta_min = 0  # 最小散射角（度）
    theta_max = 0.5  # 最大散射角（度）

    # 计算概率
    probability, integral_error = calculate_scattering_probability(
        g, theta_min, theta_max
    )

    # 输出结果
    print(f"HG相位函数数值积分计算结果：")
    print(f"不对称因子 g = {g}")
    print(f"散射角范围：{theta_min}° ~ {theta_max}°")
    print(f"散射概率：{probability:.4f}（即 {probability * 100:.1f}%）")
    print(f"积分数值误差：{integral_error:.2e}（可忽略）")
