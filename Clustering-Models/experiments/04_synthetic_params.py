"""
以高斯、螺旋、同心圆模拟数据探究 DBSCAN 的 eps、min_samples 以及 DPC 的 t0。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from src.data import make_circles_data, make_gaussian, make_spiral
from src.dpc import DensityPeakClustering
from src.metrics_util import align_labels, evaluate_clustering
from src.plotting import OUTPUT_DIR, line_plot, scatter_clusters, setup_style


DATASETS = {
    "gaussian": (make_gaussian, 3),
    "spiral": (make_spiral, 2),
    "circle": (make_circles_data, 2),
}


def k_distance(X, k):
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    dist, _ = nn.kneighbors(X)
    d = np.sort(dist[:, -1])
    return d


def sweep_dbscan(name, X, y):
    rows = []
    eps_grid = np.round(np.linspace(0.15, 1.2, 12), 3)
    ms_grid = [3, 5, 8, 12]
    for eps in eps_grid:
        for ms in ms_grid:
            pred = DBSCAN(eps=float(eps), min_samples=ms).fit_predict(X)
            m = evaluate_clustering(X, y, pred)
            m.update({"dataset": name, "eps": eps, "min_samples": ms})
            rows.append(m)
    return pd.DataFrame(rows)


def sweep_dpc(name, X, y, n_clusters):
    rows = []
    t0_grid = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]
    for t0 in t0_grid:
        pred = DensityPeakClustering(t0=t0, n_clusters=n_clusters).fit_predict(X)
        m = evaluate_clustering(X, y, pred)
        m.update({"dataset": name, "t0": t0})
        rows.append(m)
    return pd.DataFrame(rows)


def main():
    setup_style()
    out = ROOT / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)

    dbscan_all = []
    dpc_all = []

    for name, (factory, n_clusters) in DATASETS.items():
        X, y = factory()
        scatter_clusters(X, y, f"{name} 真实标签", OUTPUT_DIR / f"{name}_truth.png")

        d = k_distance(X, k=5)
        line_plot(
            list(range(len(d))),
            {"5-distance": d.tolist()},
            "按距离排序的样本",
            "到第 5 近邻的距离",
            f"{name}：k-distance 图（辅助选 eps）",
            OUTPUT_DIR / f"{name}_kdistance.png",
        )

        db = sweep_dbscan(name, X, y)
        dbscan_all.append(db)
        # 固定 min_samples=5，看 eps
        sub = db[db["min_samples"] == 5].sort_values("eps")
        line_plot(
            sub["eps"].tolist(),
            {
                "兰德系数 RI": sub["rand_index"].tolist(),
                "ARI": sub["ari"].tolist(),
                "轮廓系数": sub["silhouette"].tolist(),
            },
            "eps",
            "指标值",
            f"{name}：DBSCAN eps 的影响（min_samples=5）",
            OUTPUT_DIR / f"{name}_dbscan_eps.png",
        )
        # 选一个较优 eps 看 min_samples
        best_eps = float(sub.loc[sub["ari"].idxmax(), "eps"])
        sub2 = db[np.isclose(db["eps"], best_eps)].sort_values("min_samples")
        line_plot(
            sub2["min_samples"].tolist(),
            {
                "兰德系数 RI": sub2["rand_index"].tolist(),
                "ARI": sub2["ari"].tolist(),
                "轮廓系数": sub2["silhouette"].tolist(),
            },
            "min_samples",
            "指标值",
            f"{name}：DBSCAN min_samples 的影响（eps={best_eps}）",
            OUTPUT_DIR / f"{name}_dbscan_min_samples.png",
        )

        pred_db = DBSCAN(eps=best_eps, min_samples=5).fit_predict(X)
        scatter_clusters(
            X,
            align_labels(y, pred_db),
            f"{name} DBSCAN eps={best_eps}, min_samples=5",
            OUTPUT_DIR / f"{name}_dbscan_result.png",
            y_true=y,
        )

        dpc = sweep_dpc(name, X, y, n_clusters)
        dpc_all.append(dpc)
        line_plot(
            dpc["t0"].tolist(),
            {
                "兰德系数 RI": dpc["rand_index"].tolist(),
                "ARI": dpc["ari"].tolist(),
                "轮廓系数": dpc["silhouette"].tolist(),
            },
            "t0",
            "指标值",
            f"{name}：DPC 中 t0 的影响",
            OUTPUT_DIR / f"{name}_dpc_t0.png",
        )
        best_t0 = float(dpc.loc[dpc["ari"].idxmax(), "t0"])
        pred_dpc = DensityPeakClustering(t0=best_t0, n_clusters=n_clusters).fit_predict(X)
        scatter_clusters(
            X,
            align_labels(y, pred_dpc),
            f"{name} DPC t0={best_t0}",
            OUTPUT_DIR / f"{name}_dpc_result.png",
            y_true=y,
        )

    pd.concat(dbscan_all, ignore_index=True).to_csv(
        out / "synthetic_dbscan_sweep.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(dpc_all, ignore_index=True).to_csv(
        out / "synthetic_dpc_sweep.csv", index=False, encoding="utf-8-sig"
    )
    print("模拟数据参数实验完成，结果见 outputs/")


if __name__ == "__main__":
    main()
