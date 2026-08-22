"""指标计算、结果落盘与预测分数提取。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn import metrics


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    print(f"  保存表格: {path}")
    return path


def save_json(obj: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"  保存JSON: {path}")
    return path


def predict_scores(model, X) -> np.ndarray:
    """尽量返回正类分数，供 AUC / ROC 使用。"""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if np.ndim(proba) == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float)
        return np.asarray(proba, dtype=float).ravel()
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float).ravel()
    return np.asarray(model.predict(X), dtype=float).ravel()


def clf_metrics(y_true, y_pred, y_score=None, pos_label: int = 1) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    out = {
        "Accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "Precision": float(metrics.precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "Recall": float(metrics.recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "F1": float(metrics.f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "Precision_macro": float(metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall_macro": float(metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_macro": float(metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_score is not None:
        y_score = np.asarray(y_score).ravel()
        out["AUC"] = float(metrics.roc_auc_score(y_true, y_score))
    else:
        out["AUC"] = float("nan")
    return out


def reg_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mse = float(metrics.mean_squared_error(y_true, y_pred))
    return {
        "R2": float(metrics.r2_score(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(metrics.mean_absolute_error(y_true, y_pred)),
    }


def print_metrics(title: str, m: Dict[str, float], digits: int = 4) -> None:
    print(f"  [{title}]")
    for k, v in m.items():
        print(f"    {k}: {v:.{digits}f}")


def topk_names(names: Sequence[str], values: Sequence[float], k: int = 8) -> list:
    order = np.argsort(np.abs(np.asarray(values, dtype=float)))[::-1]
    return [names[i] for i in order[:k]]
