"""兰德系数与轮廓系数。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    rand_score,
    silhouette_score,
)


def evaluate_clustering(X, y_true, y_pred, sample_size=None):
    """
    计算兰德系数（RI）、调整兰德系数（ARI）与轮廓系数。

    轮廓系数在簇数 < 2 或全部为噪声时无法计算，返回 NaN。
    """
    y_pred = np.asarray(y_pred)
    result = {
        "rand_index": float("nan"),
        "ari": float("nan"),
        "silhouette": float("nan"),
        "n_clusters": 0,
        "n_noise": int(np.sum(y_pred == -1)),
    }

    labeled = y_pred != -1
    unique = np.unique(y_pred[labeled]) if labeled.any() else np.array([])
    result["n_clusters"] = int(len(unique))

    if y_true is not None:
        result["rand_index"] = float(rand_score(y_true, y_pred))
        result["ari"] = float(adjusted_rand_score(y_true, y_pred))

    if result["n_clusters"] >= 2 and labeled.sum() > result["n_clusters"]:
        try:
            result["silhouette"] = float(
                silhouette_score(
                    X[labeled],
                    y_pred[labeled],
                    sample_size=sample_size,
                )
            )
        except ValueError:
            result["silhouette"] = float("nan")
    return result


def align_labels(y_true, y_pred):
    """
    将簇编号按匈牙利算法对齐到真实标签，仅用于作图着色。
    兰德系数与簇编号置换无关，评估仍用原始 y_pred。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = np.full_like(y_pred, fill_value=-1)
    labeled = y_pred >= 0
    if not labeled.any():
        return y_pred
    t_u = np.unique(y_true[labeled])
    p_u = np.unique(y_pred[labeled])
    c = np.zeros((len(t_u), len(p_u)), dtype=int)
    for i, t in enumerate(t_u):
        for j, p in enumerate(p_u):
            c[i, j] = int(np.sum((y_true[labeled] == t) & (y_pred[labeled] == p)))
    ri, ci = linear_sum_assignment(-c)
    mapping = {int(p_u[j]): int(t_u[i]) for i, j in zip(ri, ci)}
    used = set(mapping.values())
    nxt = int(max(int(t_u.max()), int(p_u.max())) + 1)
    for p in p_u:
        p = int(p)
        if p not in mapping:
            while nxt in used:
                nxt += 1
            mapping[p] = nxt
            used.add(nxt)
            nxt += 1
    out[labeled] = np.array([mapping[int(v)] for v in y_pred[labeled]])
    return out
