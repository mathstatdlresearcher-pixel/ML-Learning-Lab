"""数据集加载：鸢尾花与三类模拟数据。"""

from __future__ import annotations

from . import runtime  # noqa: F401

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_blobs, make_circles
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_iris_data(scale="minmax"):
    """
    加载鸢尾花。scale:
    - False / None / 'none': 原始厘米
    - 'minmax': 各维缩放到 [0, 1]（聚类默认；保留花瓣量纲优势）
    - True / 'standard': z-score（会抬高萼片宽度权重，与真实三类更不一致）
    """
    bunch = load_iris(as_frame=True)
    X = bunch.data.to_numpy(dtype=float)
    y = bunch.target.to_numpy()
    feature_names = list(bunch.feature_names)
    df = bunch.frame.copy()
    df.columns = feature_names + ["target"]
    df["species"] = bunch.target_names[y]
    scaler = None
    if scale in (True, "standard", "zscore"):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scale in ("minmax", "min_max"):
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    return X, y, feature_names, df, scaler


def make_gaussian(n_samples=300, random_state=42):
    """高斯混合簇（球形、可分）。"""
    X, y = make_blobs(
        n_samples=n_samples,
        centers=3,
        cluster_std=[0.6, 0.8, 0.5],
        random_state=random_state,
    )
    X = StandardScaler().fit_transform(X)
    return X, y


def make_spiral(n_samples=400, noise=0.08, random_state=42):
    """双螺旋数据。"""
    rng = np.random.default_rng(random_state)
    n = n_samples // 2
    theta = np.linspace(0, 3 * np.pi, n)
    r = np.linspace(0.4, 2.2, n)
    x1 = r * np.cos(theta) + rng.normal(0, noise, n)
    y1 = r * np.sin(theta) + rng.normal(0, noise, n)
    x2 = r * np.cos(theta + np.pi) + rng.normal(0, noise, n)
    y2 = r * np.sin(theta + np.pi) + rng.normal(0, noise, n)
    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.array([0] * n + [1] * n)
    X = StandardScaler().fit_transform(X)
    return X, y


def make_circles_data(n_samples=400, noise=0.06, factor=0.45, random_state=42):
    """同心圆数据。"""
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=random_state,
    )
    X = StandardScaler().fit_transform(X)
    return X, y


def iris_dataframe_unscaled() -> pd.DataFrame:
    """未标准化的鸢尾花表，用于探索性分析。"""
    _, _, _, df, _ = load_iris_data(scale=False)
    return df
