# -*- coding: utf-8 -*-
"""
读取并加载统一流程参数配置（config.yaml）
所有医学/生物仿真主流程均应通过此入口获取全局配置，避免硬编码，便于参数自动化调整
"""
import yaml
import os


def load_config(config_path=None):
    """
    加载YAML格式的全局仿真配置信息
    :param config_path: 配置文件路径，默认为当前src目录下config.yaml
    :return: 配置参数字典
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

