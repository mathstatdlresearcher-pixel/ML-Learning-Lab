"""模拟数据：强线性信号 + 适度噪声 + 无关特征（拉开算法差距）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    DATA_DIR,
    FEATURE_SCALE,
    N_JUNK,
    N_SAMPLES,
    NOISE_RATIO,
    RANDOM_STATE,
    TEST_SIZE,
)


@dataclass
class DatasetBundle:
    name: str
    description: str
    X: pd.DataFrame
    y: pd.Series
    feature_names: List[str]
    true_coef: Dict[str, float]
    relevant_features: List[str]
    has_noise: bool = False
    multicollinear: bool = False
    noise_std: float = 0.0
    signal_std: float = 0.0


def _rng(seed: int = RANDOM_STATE) -> np.random.Generator:
    return np.random.default_rng(seed)


def _add_noise(signal: np.ndarray, ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    signal = np.asarray(signal, dtype=float)
    sig_std = float(np.std(signal))
    if ratio <= 0:
        return signal.copy(), 0.0, sig_std
    noise_std = ratio * sig_std
    y = signal + rng.normal(0.0, noise_std, size=signal.shape[0])
    return y, noise_std, sig_std


def _junk_features(n: int, p: int, rng: np.random.Generator, scale: float) -> pd.DataFrame:
    data = {f"noise{i+1}": scale * rng.standard_normal(n) for i in range(p)}
    return pd.DataFrame(data)


def generate_raw_features(n: int = N_SAMPLES, seed: int = RANDOM_STATE) -> Dict[str, np.ndarray]:
    rng = _rng(seed)
    s = float(FEATURE_SCALE)
    x1 = s * rng.random(n)
    x2 = s * rng.random(n)
    x3 = s * rng.random(n)
    x4 = 6.7 * x1 + 4 * x2 + 0.05 * s * rng.random(n)
    x5 = 2.8 * x1 + 3.4 * x4 + 0.01 * s * rng.random(n)
    return {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}


def build_simulated_datasets(n: int = N_SAMPLES, seed: int = RANDOM_STATE) -> Dict[str, DatasetBundle]:
    rng = _rng(seed)
    f = generate_raw_features(n=n, seed=seed)
    x1, x2, x3, x4, x5 = f["x1"], f["x2"], f["x3"], f["x4"], f["x5"]
    x0 = float(FEATURE_SCALE) * rng.random(n)
    junk = _junk_features(n, N_JUNK, rng, scale=float(FEATURE_SCALE))

    s1 = 0.5 + 2.0 * x1 + 0.3 * x2
    s2 = 1.0 + 2.0 * x1 + 4.0 * x2
    s3 = 5.0 + 1.0 * x0 + 1.2 * x1 + 3.5 * x2
    s4 = 0.7 * x1 + 1.1 * x2 - 6.4 * x3 + 1.0 * x4 + 1.0 * x5

    y1, n1, sig1 = _add_noise(s1, NOISE_RATIO["y1"], rng)
    y2, n2, sig2 = _add_noise(s2, NOISE_RATIO["y2"], rng)
    y3, n3, sig3 = _add_noise(s3, NOISE_RATIO["y3"], rng)
    y4, n4, sig4 = _add_noise(s4, NOISE_RATIO["y4"], rng)

    # y1: 纯线性、无噪声、无无关变量 —— 四模型都应接近完美
    X1 = pd.DataFrame({"x1": x1, "x2": x2})

    # y2: 强线性 + 轻度噪声 + 大量无关变量 —— Lasso/LARS 应明显优于 OLS
    X2 = pd.concat([pd.DataFrame({"x1": x1, "x2": x2}), junk], axis=1)
    true2 = {"intercept": 1.0, "x1": 2.0, "x2": 4.0}
    for c in junk.columns:
        true2[c] = 0.0

    # y3: 无共线、仅相关变量、轻度噪声
    X3 = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})

    # y4: 多重共线 + 无关变量 —— Ridge/Lasso/LARS 与 OLS 拉开
    X4 = pd.concat(
        [pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}), junk],
        axis=1,
    )
    true4 = {"intercept": 0.0, "x1": 0.7, "x2": 1.1, "x3": -6.4, "x4": 1.0, "x5": 1.0}
    for c in junk.columns:
        true4[c] = 0.0

    bundles = {
        "y1_no_noise": DatasetBundle(
            name="y1_no_noise",
            description="无噪声线性: y=0.5+2*x1+0.3*x2",
            X=X1,
            y=pd.Series(y1, name="y1"),
            feature_names=list(X1.columns),
            true_coef={"intercept": 0.5, "x1": 2.0, "x2": 0.3},
            relevant_features=["x1", "x2"],
            has_noise=False,
            multicollinear=False,
            noise_std=n1,
            signal_std=sig1,
        ),
        "y2_noisy": DatasetBundle(
            name="y2_noisy",
            description=f"有噪声+无关特征: y=1+2*x1+4*x2+ε, σε={n2:.3f}, junk={N_JUNK}",
            X=X2,
            y=pd.Series(y2, name="y2"),
            feature_names=list(X2.columns),
            true_coef=true2,
            relevant_features=["x1", "x2"],
            has_noise=True,
            multicollinear=False,
            noise_std=n2,
            signal_std=sig2,
        ),
        "y3_no_multicollinearity": DatasetBundle(
            name="y3_no_multicollinearity",
            description=f"无共线线性: y=5+x0+1.2*x1+3.5*x2+ε, σε={n3:.3f}",
            X=X3,
            y=pd.Series(y3, name="y3"),
            feature_names=list(X3.columns),
            true_coef={"intercept": 5.0, "x0": 1.0, "x1": 1.2, "x2": 3.5},
            relevant_features=["x0", "x1", "x2"],
            has_noise=True,
            multicollinear=False,
            noise_std=n3,
            signal_std=sig3,
        ),
        "y4_multicollinearity": DatasetBundle(
            name="y4_multicollinearity",
            description=f"多重共线+无关特征, σε={n4:.3f}, junk={N_JUNK}",
            X=X4,
            y=pd.Series(y4, name="y4"),
            feature_names=list(X4.columns),
            true_coef=true4,
            relevant_features=["x1", "x2", "x3", "x4", "x5"],
            has_noise=True,
            multicollinear=True,
            noise_std=n4,
            signal_std=sig4,
        ),
    }
    return bundles


def save_simulated_csvs(bundles: Optional[Dict[str, DatasetBundle]] = None) -> Dict[str, Path]:
    bundles = bundles or build_simulated_datasets()
    paths = {}
    for key, b in bundles.items():
        df = b.X.copy()
        df[b.y.name] = b.y.values
        path = DATA_DIR / f"{key}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        paths[key] = path
    return paths


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression

    # 只对非 noise 特征算 VIF，避免图过宽
    cols = [c for c in X.columns if not str(c).startswith("noise")]
    if len(cols) < 2:
        cols = list(X.columns)
    sub = X[cols]
    rows = []
    for col in sub.columns:
        y = sub[col].values
        Z = sub.drop(columns=[col]).values
        if Z.shape[1] == 0:
            vif = 1.0
        else:
            r2 = LinearRegression().fit(Z, y).score(Z, y)
            vif = 1.0 / max(1.0 - r2, 1e-12)
        rows.append({"feature": col, "VIF": float(vif)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def train_test_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=random_state
    )
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    for k, b in build_simulated_datasets().items():
        print(k, b.X.shape, f"noise={b.noise_std:.3f}", b.relevant_features)
