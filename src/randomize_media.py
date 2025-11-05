import random


def randomize_media(media, perturbation_bounds):
    """
    随机化介质的光学参数，使其在指定范围内波动。

    参数:
    - media: 包含组织光学属性（例如 'mua', 'mus', 'g', 'n'）的字典列表。
    - perturbation_bounds: 每个参数的扰动范围字典（例如: {'mua': 0.1, 'mus': 0.2}）。

    返回:
    - 包含随机化参数后的介质字典列表。
    """
    randomized_media = []

    for tissue in media:
        randomized_tissue = {}
        for key in tissue:
            if key in perturbation_bounds:
                # 根据扰动范围计算随机扰动值，并进行尾数截断（保留4位小数）
                perturbation = perturbation_bounds[key] * random.uniform(-1, 1)
                randomized_value = tissue[key] + perturbation

                # 确保随机化值保持在合法范围内（光学属性不能为负），且尾数截断（保留4位小数）
                randomized_value = max(0, randomized_value)
                randomized_value = round(randomized_value, 4)
                randomized_tissue[key] = randomized_value
            else:
                # 如果没有提供扰动范围，则保留原始值
                randomized_tissue[key] = tissue[key]

        randomized_media.append(randomized_tissue)

    return randomized_media


# 示例用法
if __name__ == "__main__":
    media = [
        {"mua": 0.0338, "mus": 11.9827, "g": 0.9, "n": 1.37},
        {"mua": 0.05251, "mus": 24.4153, "g": 0.9, "n": 1.37},
        {"mua": 0.0318, "mus": 15.959, "g": 0.9, "n": 1.37},
    ]

    # 定义每个参数的扰动范围
    perturbation_bounds = {
        "mua": 0.001,
        "mus": 0.02,
    }  # 保持 g 和 n 不变，仅减少 mua 和 mus 的扰动范围
    randomized = randomize_media(media, perturbation_bounds)

    print("原始介质:", media)
    print("随机化后的介质:", randomized)
