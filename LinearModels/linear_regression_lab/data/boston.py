"""Boston 房价数据加载与预处理。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import BOSTON_FEATURE_NAMES, DATA_DIR, RANDOM_STATE, TEST_SIZE


BOSTON_URLS = [
    "http://lib.stat.cmu.edu/datasets/boston",
    "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
]


def _load_from_cmu(url: str) -> pd.DataFrame:
    raw = pd.read_csv(url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw.values[::2, :], raw.values[1::2, :2]])
    target = raw.values[1::2, 2]
    df = pd.DataFrame(data, columns=BOSTON_FEATURE_NAMES)
    df["MEDV"] = target
    return df


def _load_from_csv(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    # selva86 版本列名小写
    rename = {
        "crim": "CRIM",
        "zn": "ZN",
        "indus": "INDUS",
        "chas": "CHAS",
        "nox": "NOX",
        "rm": "RM",
        "age": "AGE",
        "dis": "DIS",
        "rad": "RAD",
        "tax": "TAX",
        "ptratio": "PTRATIO",
        "b": "B",
        "lstat": "LSTAT",
        "medv": "MEDV",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "MEDV" not in df.columns and "Price" in df.columns:
        df = df.rename(columns={"Price": "MEDV"})
    return df[BOSTON_FEATURE_NAMES + ["MEDV"]]


def load_boston(cache_path: Optional[Path] = None) -> pd.DataFrame:
    """加载 Boston 房价数据（sklearn 已移除 load_boston）。"""
    cache_path = cache_path or (DATA_DIR / "boston.csv")
    if cache_path.exists():
        return pd.read_csv(cache_path)

    last_err = None
    for url in BOSTON_URLS:
        try:
            if "BostonHousing.csv" in url:
                df = _load_from_csv(url)
            else:
                df = _load_from_cmu(url)
            df.to_csv(cache_path, index=False, encoding="utf-8-sig")
            return df
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(f"无法下载 Boston 数据集，最后错误：{last_err}")


def preprocess_boston(
    df: Optional[pd.DataFrame] = None,
    drop_capped: bool = True,
    corr_threshold: float = 0.45,
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    预处理流程：
    1) 缺失检查
    2) 去掉房价封顶值 MEDV=50（常见做法）
    3) 基于与目标相关度做特征筛选
    """
    df = load_boston() if df is None else df.copy()
    info = {
        "n_raw": int(len(df)),
        "missing": df.isna().sum().to_dict(),
        "dropped_capped": 0,
        "selected_features": [],
    }

    if drop_capped and "MEDV" in df.columns:
        before = len(df)
        df = df.loc[df["MEDV"] < 50].copy()
        info["dropped_capped"] = before - len(df)

    # 简单缺失填充（本数据通常无缺失）
    if df.isna().any().any():
        df = df.fillna(df.median(numeric_only=True))

    y = df["MEDV"]
    X_all = df.drop(columns=["MEDV"])
    corr = X_all.corrwith(y).abs().sort_values(ascending=False)
    selected = corr[corr >= corr_threshold].index.tolist()
    # 保底至少保留相关性最高的 5 个特征
    if len(selected) < 5:
        selected = corr.head(5).index.tolist()
    info["selected_features"] = selected
    info["corr_with_target"] = corr.to_dict()
    return X_all[selected], y, info


def boston_train_test(
    scale: bool = True,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    X, y, info = preprocess_boston()
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=random_state
    )
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X, y, X_train, X_test, y_train, y_test, scaler, info


if __name__ == "__main__":
    df = load_boston()
    print(df.head())
    print(df.describe())
