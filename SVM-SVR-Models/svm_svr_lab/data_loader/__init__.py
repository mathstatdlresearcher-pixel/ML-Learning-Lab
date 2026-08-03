"""数据加载模块。"""

from .prepare_data import load_abalone_data, load_blobs_data, load_iris_data

__all__ = ["load_iris_data", "load_abalone_data", "load_blobs_data"]