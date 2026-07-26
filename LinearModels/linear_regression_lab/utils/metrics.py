"""评价指标工具。"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def format_metrics(metrics: Dict[str, float], digits: int = 4) -> str:
    return " | ".join(f"{k}={v:.{digits}f}" for k, v in metrics.items())


def summarize_repeats(rows: list[Dict[str, Any]], metric_keys=("R2", "MSE", "MAE", "RMSE")) -> Dict[str, Dict[str, float]]:
    """对多次重复实验结果求均值与标准差。"""
    out: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        if key not in rows[0]:
            continue
        vals = np.array([r[key] for r in rows], dtype=float)
        out[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    return out
