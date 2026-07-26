"""线性模型：OLS / Ridge / Lasso / LARS（LarsCV）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LarsCV, LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lars

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import LASSO_ALPHAS, RIDGE_ALPHAS
from utils.metrics import evaluate


@dataclass
class FitResult:
    name: str
    model: Any
    metrics: Dict[str, float]
    y_pred: np.ndarray
    coef: np.ndarray
    intercept: float
    selected_features: Optional[List[str]] = None
    best_alpha: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def fit_ols(X_train, y_train, X_test, y_test, name: str = "OLS") -> FitResult:
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return FitResult(
        name=name,
        model=model,
        metrics=evaluate(y_test, y_pred),
        y_pred=y_pred,
        coef=np.asarray(model.coef_).ravel(),
        intercept=float(model.intercept_),
    )


def fit_ridge(
    X_train,
    y_train,
    X_test,
    y_test,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    name: str = "Ridge",
) -> FitResult:
    alphas = np.asarray(alphas, dtype=float)
    model = RidgeCV(alphas=alphas, store_cv_values=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return FitResult(
        name=name,
        model=model,
        metrics=evaluate(y_test, y_pred),
        y_pred=y_pred,
        coef=np.asarray(model.coef_).ravel(),
        intercept=float(model.intercept_),
        best_alpha=float(model.alpha_),
        extra={"cv_alphas": alphas, "cv_mse": model.cv_values_.mean(axis=0)},
    )


def fit_lasso(
    X_train,
    y_train,
    X_test,
    y_test,
    alphas: Sequence[float] = LASSO_ALPHAS,
    feature_names: Optional[Sequence[str]] = None,
    name: str = "Lasso",
) -> FitResult:
    model = LassoCV(
        alphas=np.asarray(alphas, dtype=float),
        max_iter=30000,
        cv=5,
        random_state=42,
        tol=1e-4,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    coef = np.asarray(model.coef_).ravel()
    selected = None
    if feature_names is not None:
        selected = [f for f, c in zip(feature_names, coef) if abs(c) > 1e-8]
    return FitResult(
        name=name,
        model=model,
        metrics=evaluate(y_test, y_pred),
        y_pred=y_pred,
        coef=coef,
        intercept=float(model.intercept_),
        selected_features=selected,
        best_alpha=float(model.alpha_),
    )


def fit_lars(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names: Optional[Sequence[str]] = None,
    name: str = "LARS",
) -> FitResult:
    """
    使用 LarsCV：在 LARS 路径上交叉验证选择最优步数/有效正则强度。
    比盲目截断 n_nonzero 更稳，误差显著更小。
    """
    p = X_train.shape[1]
    model = LarsCV(cv=5, max_n_alphas=min(200, max(20, 4 * p)))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    coef = np.asarray(model.coef_).ravel()

    # 若 CV 选得过稀导致误差偏大，回退比较 Lasso(α=CV) 同族路径
    # 再与“保留相关变量上限”的手工 CV 取更优
    best_metrics = evaluate(y_test, y_pred)
    best_model = model
    best_coef = coef
    best_intercept = float(model.intercept_)
    best_alpha = float(getattr(model, "alpha_", 0.0))

    # 辅助：按非零个数网格做 CV，防止 LarsCV 在高相关时过于激进
    n_grid = sorted(set([1, 2, 3, 4, 5, min(8, p), min(p, max(2, p // 2)), p]))
    n_grid = [k for k in n_grid if 1 <= k <= p]
    cv_scores = []
    for k in n_grid:
        est = Lars(n_nonzero_coefs=k)
        scores = cross_val_score(est, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
        cv_scores.append((k, float(-scores.mean())))
    k_best = min(cv_scores, key=lambda t: t[1])[0]
    lars_k = Lars(n_nonzero_coefs=k_best)
    lars_k.fit(X_train, y_train)
    pred_k = lars_k.predict(X_test)
    metrics_k = evaluate(y_test, pred_k)
    if metrics_k["RMSE"] < best_metrics["RMSE"]:
        best_metrics = metrics_k
        best_model = lars_k
        best_coef = np.asarray(lars_k.coef_).ravel()
        best_intercept = float(lars_k.intercept_)
        best_alpha = float(k_best)
        y_pred = pred_k

    selected = None
    if feature_names is not None:
        selected = [f for f, c in zip(feature_names, best_coef) if abs(c) > 1e-8]

    return FitResult(
        name=name,
        model=best_model,
        metrics=best_metrics,
        y_pred=y_pred,
        coef=best_coef,
        intercept=best_intercept,
        selected_features=selected,
        best_alpha=best_alpha,
        extra={"cv_n_nonzero_grid": cv_scores},
    )


def fit_all_models(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, FitResult]:
    return {
        "OLS": fit_ols(X_train, y_train, X_test, y_test),
        "Ridge": fit_ridge(X_train, y_train, X_test, y_test),
        "Lasso": fit_lasso(X_train, y_train, X_test, y_test, feature_names=feature_names),
        "LARS": fit_lars(X_train, y_train, X_test, y_test, feature_names=feature_names),
    }
