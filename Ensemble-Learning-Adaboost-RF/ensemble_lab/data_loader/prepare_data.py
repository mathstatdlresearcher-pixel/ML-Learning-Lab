"""读取本地 CSV，清洗列名，划分训练/测试集。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common.config import (
    BOSTON_CSV,
    BREAST_CSV,
    DATA_OUT_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from common.utils import save_csv, save_json


@dataclass
class SplitBundle:
    feature_names: list
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    scaler: StandardScaler
    raw_df: pd.DataFrame


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_breast_raw() -> pd.DataFrame:
    if not BREAST_CSV.exists():
        raise FileNotFoundError(f"找不到乳腺癌数据: {BREAST_CSV}")
    df = _clean_columns(pd.read_csv(BREAST_CSV))
    if "diagnosis" not in df.columns:
        raise ValueError("breast_cancer.csv 缺少 diagnosis 列")
    # 原始 0=恶性, 1=良性；改为 1=恶性，便于 Precision/Recall 关注阳性类
    df["diagnosis_raw"] = df["diagnosis"].astype(int)
    df["diagnosis"] = (df["diagnosis_raw"] == 0).astype(int)
    return df


def load_boston_raw() -> pd.DataFrame:
    if not BOSTON_CSV.exists():
        raise FileNotFoundError(f"找不到 Boston 数据: {BOSTON_CSV}")
    df = _clean_columns(pd.read_csv(BOSTON_CSV))
    if "target" in df.columns and "MEDV" not in df.columns:
        df = df.rename(columns={"target": "MEDV"})
    if "MEDV" not in df.columns:
        raise ValueError("boston.csv 缺少 target/MEDV 列")
    return df


def _iqr_clip(X: pd.DataFrame, factor: float = 3.0) -> Tuple[pd.DataFrame, dict]:
    """温和截尾，避免极端值主导线性基学习器。"""
    info = {}
    out = X.copy()
    for col in out.columns:
        q1, q3 = out[col].quantile(0.25), out[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        before = int(((out[col] < lo) | (out[col] > hi)).sum())
        out[col] = out[col].clip(lo, hi)
        info[col] = {"n_clipped": before, "lo": float(lo), "hi": float(hi)}
    return out, info


def prepare_breast(test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> SplitBundle:
    df = load_breast_raw()
    y = df["diagnosis"].to_numpy()
    X = df.drop(columns=["diagnosis", "diagnosis_raw"])
    X, clip_info = _iqr_clip(X)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(dtype=float),
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    save_csv(pd.DataFrame(X_train, columns=feature_names), DATA_OUT_DIR / "breast_X_train.csv")
    save_csv(pd.DataFrame(X_test, columns=feature_names), DATA_OUT_DIR / "breast_X_test.csv")
    save_csv(pd.DataFrame({"y": y_train}), DATA_OUT_DIR / "breast_y_train.csv")
    save_csv(pd.DataFrame({"y": y_test}), DATA_OUT_DIR / "breast_y_test.csv")
    joblib.dump({"scaler": scaler, "feature_names": feature_names}, MODEL_DIR / "breast_preprocess.joblib")
    save_json(
        {
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_features": len(feature_names),
            "pos_rate_train": float(y_train.mean()),
            "pos_rate_test": float(y_test.mean()),
            "clip": clip_info,
            "note": "diagnosis: 1=恶性, 0=良性",
        },
        DATA_OUT_DIR / "breast_split_info.json",
    )
    return SplitBundle(
        feature_names=feature_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        scaler=scaler,
        raw_df=df,
    )


def prepare_boston(
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    drop_capped: bool = False,
) -> SplitBundle:
    df = load_boston_raw()
    info = {"n_raw": int(len(df)), "n_capped50": int((df["MEDV"] >= 50).sum()), "drop_capped": drop_capped}
    if drop_capped:
        df = df.loc[df["MEDV"] < 50].copy()
    if df.isna().any().any():
        df = df.fillna(df.median(numeric_only=True))

    y = df["MEDV"].to_numpy(dtype=float)
    X = df.drop(columns=["MEDV"])
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(dtype=float),
        y,
        test_size=test_size,
        random_state=random_state,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    save_csv(pd.DataFrame(X_train, columns=feature_names), DATA_OUT_DIR / "boston_X_train.csv")
    save_csv(pd.DataFrame(X_test, columns=feature_names), DATA_OUT_DIR / "boston_X_test.csv")
    save_csv(pd.DataFrame({"y": y_train}), DATA_OUT_DIR / "boston_y_train.csv")
    save_csv(pd.DataFrame({"y": y_test}), DATA_OUT_DIR / "boston_y_test.csv")
    joblib.dump({"scaler": scaler, "feature_names": feature_names}, MODEL_DIR / "boston_preprocess.joblib")
    save_json(
        {
            **info,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_features": len(feature_names),
            "y_mean_train": float(y_train.mean()),
            "y_mean_test": float(y_test.mean()),
        },
        DATA_OUT_DIR / "boston_split_info.json",
    )
    return SplitBundle(
        feature_names=feature_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        scaler=scaler,
        raw_df=df,
    )


def get_breast(force: bool = False) -> SplitBundle:
    marker = DATA_OUT_DIR / "breast_split_info.json"
    if force or not marker.exists():
        return prepare_breast()
    blob = joblib.load(MODEL_DIR / "breast_preprocess.joblib")
    feature_names = blob["feature_names"]
    scaler: StandardScaler = blob["scaler"]
    X_train = pd.read_csv(DATA_OUT_DIR / "breast_X_train.csv").to_numpy(dtype=float)
    X_test = pd.read_csv(DATA_OUT_DIR / "breast_X_test.csv").to_numpy(dtype=float)
    y_train = pd.read_csv(DATA_OUT_DIR / "breast_y_train.csv")["y"].to_numpy()
    y_test = pd.read_csv(DATA_OUT_DIR / "breast_y_test.csv")["y"].to_numpy()
    return SplitBundle(
        feature_names=feature_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_scaled=scaler.transform(X_train),
        X_test_scaled=scaler.transform(X_test),
        scaler=scaler,
        raw_df=load_breast_raw(),
    )


def get_boston(force: bool = False) -> SplitBundle:
    marker = DATA_OUT_DIR / "boston_split_info.json"
    if force or not marker.exists():
        return prepare_boston()
    blob = joblib.load(MODEL_DIR / "boston_preprocess.joblib")
    feature_names = blob["feature_names"]
    scaler: StandardScaler = blob["scaler"]
    X_train = pd.read_csv(DATA_OUT_DIR / "boston_X_train.csv").to_numpy(dtype=float)
    X_test = pd.read_csv(DATA_OUT_DIR / "boston_X_test.csv").to_numpy(dtype=float)
    y_train = pd.read_csv(DATA_OUT_DIR / "boston_y_train.csv")["y"].to_numpy(dtype=float)
    y_test = pd.read_csv(DATA_OUT_DIR / "boston_y_test.csv")["y"].to_numpy(dtype=float)
    return SplitBundle(
        feature_names=feature_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_scaled=scaler.transform(X_train),
        X_test_scaled=scaler.transform(X_test),
        scaler=scaler,
        raw_df=load_boston_raw(),
    )
