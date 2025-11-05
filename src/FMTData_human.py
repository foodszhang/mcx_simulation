import pmcx
import numpy as np
import torch
import scipy.io as sio
from datetime import datetime
import json
import os
import time
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from skimage import morphology, draw
from scipy import ndimage


class NIRIIFMTDataGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.config = self._default_config()
        print(f"使用设备: {device}")
        print("NIR-II区荧光分子断层成像数据生成器已初始化")

    def _default_config(self):
        return {
            'volume_size': [80, 80, 80],  # 减小体模尺寸以提高计算效率#######################
            'voxel_size': 0.2,  # 适当增加体素大小#######################
            'n_photons': 50000000,  # 减少光子数以提高计算速度######################
            'time_window': [0, 3e-9],  # 调整时间窗口#######################
            'time_step': 3e-9,  # 增加时间步长##############会有影响
            'source_position': [40, 40, 0],  # 调整光源位置#####################
            'source_direction': [0, 0, 1],  # 光源方向不变
            'optical_properties': {
                # NIR-II区光学属性 (μa, μs, g, n)
                'background_ex': [0.002, 0.8, 0.85, 1.38],  # 背景激发属性
                'target_ex': [0.015, 1.0, 0.82, 1.38],  # 目标激发属性
                'skin_ex': [0.003, 0.9, 0.83, 1.38],  # 皮肤组织激发属性
                'fat_ex': [0.0015, 0.7, 0.8, 1.38],  # 脂肪组织激发属性
                'vessel_ex': [0.025, 1.1, 0.85, 1.38],  # 血管组织激发属性
                'background_em': [0.0015, 0.75, 0.85, 1.38],  # 背景发射属性
                'target_em': [0.012, 0.9, 0.82, 1.38],  # 目标发射属性
                'skin_em': [0.0025, 0.85, 0.83, 1.38],  # 皮肤组织发射属性
                'fat_em': [0.0012, 0.65, 0.8, 1.38],  # 脂肪组织发射属性
                'vessel_em': [0.02, 1.0, 0.85, 1.38]  # 血管组织发射属性
            },
            'fluorophore_properties': {
                'quantum_yield': 0.08,  # 量子产率
                'lifetime': 0.8e-9,  # 荧光寿命
                'excitation_wavelength': 1064,  # 激发波长
                'emission_wavelength': 1300  # 发射波长
            },
            'noise_model': {
                'poisson_scale': 0.0005,  # 泊松噪声强度
                'gaussian_sigma': 0.0005,  # 高斯噪声
                'background_offset': 0.00005  # 背景偏移
            },
            'detector_config': {
                'num_detectors': 24,  # 减少探测器数量###################
                'detector_radius': 1.5,  # 增加探测器半径###################
                'detector_positions': None
            },
            'normalization': {
                'method': 'log1p',
                'epsilon': 1e-12
            },
            'reconstruction_parameters': {
                'regularization_type': 'L2',
                'regularization_strength': 0.1,
                'iterations': 100
            }
        }

    def normalize_vector(self, v):
        """将向量归一化为单位向量"""
        v_array = np.array(v, dtype=np.float32)
        norm = np.linalg.norm(v_array)
        if norm == 0:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return v_array / norm

    def calculate_detector_positions(self):
        """计算探测器位置"""
        num_detectors = self.config['detector_config']['num_detectors']
        radius = self.config['detector_config']['detector_radius']
        volume_size = self.config['volume_size']

        # 计算探测器环的半径（基于体模尺寸）
        detector_ring_radius = min(volume_size[0], volume_size[1]) * 0.4

        angles = np.linspace(0, 2 * np.pi, num_detectors, endpoint=False)
        detector_positions = []

        for angle in angles:
            x = volume_size[0] // 2 + detector_ring_radius * np.cos(angle)
            y = volume_size[1] // 2 + detector_ring_radius * np.sin(angle)
            z = volume_size[2] - 2  # 探测器位于体模顶部附近
            detector_positions.append([x, y, z])

        self.config['detector_config']['detector_positions'] = detector_positions
        return detector_positions

    def create_complex_phantom(self, num_targets=1, min_radius=2, max_radius=5):
        """创建更符合生物组织特性的体模"""
        volume_size = self.config['volume_size']
        volume = np.ones(volume_size, dtype='uint8')  # 背景组织 (标签1)
        fluorophore_map = np.zeros(volume_size, dtype='float32')

        # 添加组织层结构 - 模拟皮肤、脂肪和肌肉层
        skin_depth = 2  # 皮肤层深度（体素）
        fat_depth = 6  # 脂肪层深度（体素）

        # 皮肤层 (标签2)
        volume[:, :, :skin_depth] = 2

        # 脂肪层 (标签3)
        volume[:, :, skin_depth:skin_depth + fat_depth] = 3

        # 添加背景异质性 - 模拟血管和组织不均匀性 (标签4)
        for _ in range(3):  # 减少血管数量
            pos = [
                random.randint(5, volume_size[0] - 5),
                random.randint(5, volume_size[1] - 5),
                random.randint(skin_depth + fat_depth, volume_size[2] - 5)
            ]
            radius = random.randint(1, 3)  # 减小血管半径
            length = random.randint(5, 10)  # 减小血管长度

            # 创建管状结构模拟血管
            axis = random.randint(0, 2)
            if axis == 0:  # x轴方向
                z, y, x = np.ogrid[:volume_size[0], :volume_size[1], :volume_size[2]]
                mask = ((y - pos[1]) ** 2 + (z - pos[2]) ** 2 <= radius ** 2) & \
                       (abs(x - pos[0]) <= length / 2)
            elif axis == 1:  # y轴方向
                z, y, x = np.ogrid[:volume_size[0], :volume_size[1], :volume_size[2]]
                mask = ((x - pos[0]) ** 2 + (z - pos[2]) ** 2 <= radius ** 2) & \
                       (abs(y - pos[1]) <= length / 2)
            else:  # z轴方向
                z, y, x = np.ogrid[:volume_size[0], :volume_size[1], :volume_size[2]]
                mask = ((x - pos[0]) ** 2 + (y - pos[1]) ** 2 <= radius ** 2) & \
                       (abs(z - pos[2]) <= length / 2)

            volume[mask] = 4  # 血管组织

        target_positions = []
        target_radii = []
        target_shapes = []

        for _ in range(num_targets):
            shape_type = random.randint(0, 3)
            radius = random.randint(min_radius, max_radius)

            # 确保目标位于肌肉层内
            pos = [
                random.randint(radius, volume_size[0] - radius - 1),
                random.randint(radius, volume_size[1] - radius - 1),
                random.randint(skin_depth + fat_depth + radius, volume_size[2] - radius - 1)
            ]

            # 根据形状类型创建目标 (标签5)
            if shape_type == 0:  # 球形
                z, y, x = np.ogrid[:volume_size[0], :volume_size[1], :volume_size[2]]
                distance = np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2 + (z - pos[2]) ** 2)
                mask = distance <= radius
                target_shapes.append("sphere")

            elif shape_type == 1:  # 椭球形
                z, y, x = np.ogrid[:volume_size[0], :volume_size[1], :volume_size[2]]
                rx = radius
                ry = random.randint(int(radius * 0.7), int(radius * 1.3))
                rz = random.randint(int(radius * 0.7), int(radius * 1.3))
                mask = ((x - pos[0]) ** 2 / rx ** 2 +
                        (y - pos[1]) ** 2 / ry ** 2 +
                        (z - pos[2]) ** 2 / rz ** 2) <= 1
                target_shapes.append("ellipsoid")

            elif shape_type == 2:  # 圆柱形
                height = random.randint(int(radius * 1.5), int(radius * 3))
                axis = random.randint(0, 2)

                if axis == 0:
                    z, y, x = np.ogrid[:volume_size[0], :volume_size[1], :volume_size[2]]
                    mask = ((y - pos[1]) ** 2 + (z - pos[2]) ** 2 <= radius ** 2) & \
                           (abs(x - pos[0]) <= height / 2)
                elif axis == 1:
                    z, y, x = np.ogrid[:volume_size[0], :volume_size[1], :volume_size[2]]
                    mask = ((x - pos[0]) ** 2 + (z - pos[2]) ** 2 <= radius ** 2) & \
                           (abs(y - pos[1]) <= height / 2)
                else:
                    z, y, x = np.ogrid[:volume_size[0], :volume_size[1], :volume_size[2]]
                    mask = ((x - pos[0]) ** 2 + (y - pos[1]) ** 2 <= radius ** 2) & \
                           (abs(z - pos[2]) <= height / 2)
                target_shapes.append("cylinder")

            else:  # 不规则形状
                irr_radius = random.randint(min_radius, max_radius)
                z, y, x = np.ogrid[:volume_size[0], :volume_size[1], :volume_size[2]]
                mask = np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2 + (z - pos[2]) ** 2) <= irr_radius

                for _ in range(random.randint(2, 5)):
                    offset = [
                        random.randint(-irr_radius // 2, irr_radius // 2),
                        random.randint(-irr_radius // 2, irr_radius // 2),
                        random.randint(-irr_radius // 2, irr_radius // 2)
                    ]
                    bump_pos = [p + o for p, o in zip(pos, offset)]
                    bump_radius = random.randint(1, irr_radius // 2)

                    z, y, x = np.ogrid[:volume_size[0], :volume_size[1], :volume_size[2]]
                    bump_mask = np.sqrt((x - bump_pos[0]) ** 2 +
                                        (y - bump_pos[1]) ** 2 +
                                        (z - bump_pos[2]) ** 2) <= bump_radius
                    mask = mask | bump_mask

                target_shapes.append("irregular")

            volume[mask] = 5  # 标记为目标组织

            # 使用更符合生物分布的浓度梯度
            concentration = random.uniform(0.2, 0.8)

            if shape_type == 0:
                z, y, x = np.ogrid[:volume_size[0], :volume_size[1], :volume_size[2]]
                distance = np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2 + (z - pos[2]) ** 2)
                normalized_dist = distance / radius
                # 使用指数衰减模拟荧光团分布
                gradient = np.exp(-2 * normalized_dist)
                fluorophore_map[mask] = concentration * gradient[mask]
            else:
                # 对于非球形目标，添加轻度不均匀性
                z, y, x = np.ogrid[:volume_size[0], :volume_size[1], :volume_size[2]]
                center = np.array([pos[0], pos[1], pos[2]])
                points = np.array(np.where(mask)).T
                distances = np.linalg.norm(points - center, axis=1)
                max_dist = np.max(distances)
                normalized_dists = distances / max_dist
                gradients = np.exp(-1.5 * normalized_dists)
                fluorophore_map[mask] = concentration * gradients

            target_positions.append(pos)
            target_radii.append(radius)

        return volume, fluorophore_map, target_positions, target_radii, target_shapes

    def run_excitation_simulation(self, volume):
        try:
            # 根据组织类型设置光学属性
            # 注意: 标签0通常不使用，但必须存在
            # 标签1: 背景组织
            # 标签2: 皮肤组织
            # 标签3: 脂肪组织
            # 标签4: 血管组织
            # 标签5: 目标组织
            prop = np.array([
                [0, 0, 1, 1],  # 介质0 (通常不使用)
                self.config['optical_properties']['background_ex'],  # 标签1: 背景组织
                self.config['optical_properties']['skin_ex'],  # 标签2: 皮肤组织
                self.config['optical_properties']['fat_ex'],  # 标签3: 脂肪组织
                self.config['optical_properties']['vessel_ex'],  # 标签4: 血管组织
                self.config['optical_properties']['target_ex']  # 标签5: 目标组织
            ], dtype=np.float32)

            srcdir = self.normalize_vector(self.config['source_direction'])

            result = pmcx.run(
                nphoton=self.config['n_photons'],
                vol=volume,
                tstart=self.config['time_window'][0],
                tend=self.config['time_window'][1],
                tstep=self.config['time_step'],
                srcpos=self.config['source_position'],
                srcdir=srcdir.tolist(),
                prop=prop,
                issrcfrom0=1,
                outputtype='flux'
            )
            return result

        except Exception as e:
            print(f"激发光模拟错误: {str(e)}")
            return None

    def run_emission_simulation(self, volume, excitation_flux, fluorophore_map):
        """改进的发射光模拟，考虑荧光寿命和量子产率"""
        try:
            # 处理激发光通量数据
            excitation_flux = self.process_flux_data(excitation_flux)

            # 确保形状匹配
            if excitation_flux.shape != volume.shape:
                print(f"激发光通量形状: {excitation_flux.shape}, 体模形状: {volume.shape}")
                # 尝试调整形状
                if len(excitation_flux.shape) == 4 and excitation_flux.shape[0] == 1:
                    excitation_flux = excitation_flux[0]
                elif len(excitation_flux.shape) == 4 and excitation_flux.shape[-1] == 1:
                    excitation_flux = excitation_flux[..., 0]
                else:
                    excitation_flux = np.mean(excitation_flux, axis=0)

                print(f"调整后的激发光通量形状: {excitation_flux.shape}")

                # 如果形状仍然不匹配，尝试调整大小
                if excitation_flux.shape != volume.shape:
                    print(f"形状仍然不匹配，尝试调整大小")
                    excitation_flux = np.resize(excitation_flux, volume.shape)

            excitation_flux = np.clip(excitation_flux, 0, None)
            fluorophore_map = np.clip(fluorophore_map, 0, 1)

            quantum_yield = self.config['fluorophore_properties']['quantum_yield']

            # 计算荧光源
            fluor_source = excitation_flux * fluorophore_map * quantum_yield

            if np.sum(fluor_source) > 0:
                fluor_source_normalized = fluor_source / np.sum(fluor_source)
            else:
                fluor_source_normalized = np.ones_like(fluor_source) / np.prod(fluor_source.shape)

            # 发射光的光学属性
            prop_em = np.array([
                [0, 0, 1, 1],  # 介质0 (通常不使用)
                self.config['optical_properties']['background_em'],  # 标签1: 背景组织
                self.config['optical_properties']['skin_em'],  # 标签2: 皮肤组织
                self.config['optical_properties']['fat_em'],  # 标签3: 脂肪组织
                self.config['optical_properties']['vessel_em'],  # 标签4: 血管组织
                self.config['optical_properties']['target_em']  # 标签5: 目标组织
            ], dtype=np.float32)

            default_srcdir = self.normalize_vector([0, 0, 1])

            result = pmcx.run(
                nphoton=int(self.config['n_photons'] * 0.8),
                vol=volume,
                tstart=self.config['time_window'][0],
                tend=self.config['time_window'][1],
                tstep=self.config['time_step'],
                srcpattern=fluor_source_normalized,
                srcdir=default_srcdir.tolist(),
                prop=prop_em,
                issrcfrom0=1,
                outputtype='flux',
                maxdetphoton=500000,
                maxjumpdebug=5000
            )
            return result

        except Exception as e:
            print(f"发射光模拟错误: {str(e)}")
            return None

    def normalize_data(self, data, method='log1p'):
        """标准化数据"""
        epsilon = self.config['normalization']['epsilon']

        if method == 'log1p':
            return np.log1p(data)
        elif method == 'zscore':
            mean = np.mean(data)
            std = np.std(data)
            return (data - mean) / (std + epsilon)
        elif method == 'minmax':
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max - data_min > epsilon:
                return (data - data_min) / (data_max - data_min)
            else:
                return np.zeros_like(data)
        else:
            return data

    def add_noise(self, data):
        """改进的噪声模型，考虑NIR-II区特性"""
        noise_config = self.config['noise_model']
        epsilon = self.config['normalization']['epsilon']

        normalized_data = self.normalize_data(data, 'log1p')

        # 添加适度的泊松噪声
        poisson_noise = np.random.poisson(noise_config['poisson_scale'] * normalized_data)
        noisy_data = normalized_data + poisson_noise

        # 添加高斯噪声
        gaussian_noise = np.random.normal(0, noise_config['gaussian_sigma'], noisy_data.shape)
        noisy_data = noisy_data + gaussian_noise

        # 添加背景偏移
        noisy_data = noisy_data + noise_config['background_offset']

        # 确保非负
        noisy_data = np.clip(noisy_data, 0, None)

        return noisy_data

    def process_flux_data(self, flux_data):
        """处理通量数据"""
        # 如果flux_data是字典，尝试提取'flux'字段
        if isinstance(flux_data, dict) and 'flux' in flux_data:
            flux_data = flux_data['flux']

        # 处理不同形状的数据
        if len(flux_data.shape) == 4:
            # 4D数据: 可能是[时间, 深度, 高度, 宽度]或[深度, 高度, 宽度, 时间]
            if flux_data.shape[0] == 1:
                flux_data = flux_data[0]  # 去除时间维度
            elif flux_data.shape[-1] == 1:
                flux_data = flux_data[..., 0]  # 去除时间维度
            else:
                # 对时间维度取平均
                flux_data = np.mean(flux_data, axis=0)
        elif len(flux_data.shape) == 3:
            # 已经是3D数据，直接使用
            pass
        else:
            # 未知形状，尝试转换为3D
            print(f"警告: 未知的通量数据形状: {flux_data.shape}")
            if flux_data.size == np.prod(self.config['volume_size']):
                flux_data = flux_data.reshape(self.config['volume_size'])
            else:
                print(f"无法处理通量数据形状: {flux_data.shape}")
                return None

        return flux_data

    def simulate_detector_readings(self, volume, flux_data):
        """改进的探测器读数计算"""
        detector_positions = self.config['detector_config']['detector_positions']
        if detector_positions is None:
            self.calculate_detector_positions()
            detector_positions = self.config['detector_config']['detector_positions']

        detector_radius = self.config['detector_config']['detector_radius']
        readings = []
        epsilon = self.config['normalization']['epsilon']

        for pos in detector_positions:
            z, y, x = np.ogrid[:volume.shape[0], :volume.shape[1], :volume.shape[2]]
            distance = np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2 + (z - pos[2]) ** 2)
            detector_mask = distance <= detector_radius

            if np.any(detector_mask):
                # 使用加权求和，考虑距离衰减
                weights = np.exp(-distance[detector_mask] / 2.0)  # 距离衰减权重
                detector_reading = np.sum(flux_data[detector_mask] * weights) + epsilon
                detector_reading = self.normalize_data(detector_reading, 'log1p')
            else:
                detector_reading = 0
            readings.append(detector_reading)

        return np.array(readings)

    def generate_single_sample(self, sample_idx, output_dir):
        """生成单个样本数据"""
        try:
            num_targets = random.randint(1, 3)  # 减少目标数量，更符合实际情况

            volume, fluorophore_map, target_positions, target_radii, target_shapes = \
                self.create_complex_phantom(num_targets)

            self.calculate_detector_positions()

            # 运行激发光模拟
            excitation_result = self.run_excitation_simulation(volume)
            if excitation_result is None:
                print(f"样本 {sample_idx} 激发光模拟失败")
                return False

            excitation_flux = excitation_result

            # 运行发射光模拟
            emission_result = self.run_emission_simulation(volume, excitation_flux, fluorophore_map)
            if emission_result is None:
                print(f"样本 {sample_idx} 发射光模拟失败")
                return False

            emission_flux = emission_result

            # 处理通量数据
            processed_excitation_flux = self.process_flux_data(excitation_flux)
            processed_emission_flux = self.process_flux_data(emission_flux)

            if processed_excitation_flux is None or processed_emission_flux is None:
                print(f"样本 {sample_idx} 通量数据处理失败")
                return False

            # 应用标准化
            normalized_excitation_flux = self.normalize_data(processed_excitation_flux)
            normalized_emission_flux = self.normalize_data(processed_emission_flux)
            normalized_fluorophore_map = self.normalize_data(fluorophore_map)

            # 添加噪声
            noisy_excitation_flux = self.add_noise(normalized_excitation_flux)
            noisy_emission_flux = self.add_noise(normalized_emission_flux)

            # 模拟探测器读数
            detector_readings_ex = self.simulate_detector_readings(volume, noisy_excitation_flux)
            detector_readings_em = self.simulate_detector_readings(volume, noisy_emission_flux)

            # 准备数据字典
            data_dict = {
                'excitation_flux': normalized_excitation_flux.astype(np.float32),
                'emission_flux': normalized_emission_flux.astype(np.float32),
                'noisy_excitation_flux': noisy_excitation_flux.astype(np.float32),
                'noisy_emission_flux': noisy_emission_flux.astype(np.float32),
                'detector_readings_ex': detector_readings_ex.astype(np.float32),
                'detector_readings_em': detector_readings_em.astype(np.float32),
                'fluorophore_map': normalized_fluorophore_map.astype(np.float32),
                'volume': volume.astype(np.uint8),
                'config': self.config,
                'target_positions': np.array(target_positions, dtype=np.float32),
                'target_radii': np.array(target_radii, dtype=np.float32),
                'target_shapes': target_shapes,
                'simulation_time': datetime.now().isoformat(),
                'sample_id': sample_idx
            }

            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 保存数据
            save_path = os.path.join(output_dir, f"sample_{sample_idx:04d}.mat")
            sio.savemat(save_path, data_dict)

            print(f"样本 {sample_idx} 生成成功")
            return True

        except Exception as e:
            print(f"生成样本 {sample_idx} 时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def generate_batch_samples(self, num_samples=100, output_dir="./nirii_fmt_dataset"):
        """批量生成样本数据"""
        os.makedirs(output_dir, exist_ok=True)

        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

        success_count = 0
        start_time = time.time()

        print(f"开始生成 {num_samples} 个NIR-II区FMT样本...")

        for i in tqdm(range(num_samples), desc="生成样本"):
            success = self.generate_single_sample(i, output_dir)
            if success:
                success_count += 1

        end_time = time.time()
        total_time = end_time - start_time

        stats = {
            "total_samples": num_samples,
            "successful_samples": success_count,
            "failed_samples": num_samples - success_count,
            "success_rate": success_count / num_samples,
            "total_time_seconds": total_time,
            "average_time_per_sample": total_time / num_samples,
            "completion_time": datetime.now().isoformat()
        }

        stats_path = os.path.join(output_dir, "stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)

        print(f"NIR-II区FMT样本生成完成!")
        print(f"成功: {success_count}/{num_samples} ({(success_count / num_samples) * 100:.2f}%)")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均每个样本耗时: {total_time / num_samples:.2f} 秒")

        return stats


# 使用示例
if __name__ == "__main__":
    generator = NIRIIFMTDataGenerator(device='cuda')

    # 生成小批量样本进行测试
    stats = generator.generate_batch_samples(
        num_samples=5,  # 生成5个样本测试
        output_dir="./nirii_fmt_dataset"
    )

    # 只有在有成功样本时才尝试加载
    if stats['successful_samples'] > 0:
        sample_path = os.path.join("./nirii_fmt_dataset", f"sample_0000.mat")
        if os.path.exists(sample_path):
            sample_data = sio.loadmat(sample_path)
            print("\n数据范围验证:")
            print(
                f"激发光通量范围: [{np.min(sample_data['excitation_flux'])}, {np.max(sample_data['excitation_flux'])}]")
            print(f"发射光通量范围: [{np.min(sample_data['emission_flux'])}, {np.max(sample_data['emission_flux'])}]")
            print(
                f"探测器读数范围: [{np.min(sample_data['detector_readings_em'])}, {np.max(sample_data['detector_readings_em'])}]")
            print(
                f"荧光团分布范围: [{np.min(sample_data['fluorophore_map'])}, {np.max(sample_data['fluorophore_map'])}]")
        else:
            print(f"\n警告: 样本文件 {sample_path} 不存在")
    else:
        print("\n没有成功生成任何样本，无法验证数据范围")