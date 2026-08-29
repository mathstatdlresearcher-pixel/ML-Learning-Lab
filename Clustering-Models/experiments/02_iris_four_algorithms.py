"""2. 四种算法在鸢尾花上的性能（兰德系数、轮廓系数）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import runtime  # noqa: F401

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans

from src.data import load_iris_data
from src.dpc import DensityPeakClustering
from src.metrics_util import align_labels, evaluate_clustering
from src.plotting import OUTPUT_DIR, scatter_clusters, setup_style


def main():
    setup_style()
    X, y, names, _, _ = load_iris_data(scale="minmax")
    # 花瓣长宽最能分开三类，作图用这两维（聚类仍用全部 4 维）
    petal = X[:, 2:4]

    models = {
        "K-Means": KMeans(n_clusters=3, n_init=10, random_state=42),
        "层次聚类": AgglomerativeClustering(n_clusters=3, linkage="ward"),
        "DBSCAN": DBSCAN(eps=0.13, min_samples=5),
        "DPC": DensityPeakClustering(t0=0.08, n_clusters=3),
    }

    rows = []
    for name, model in models.items():
        pred = model.fit_predict(X)
        m = evaluate_clustering(X, y, pred)
        m["algorithm"] = name
        rows.append(m)
        vis = align_labels(y, pred)
        fig_name = {"层次聚类": "Hierarchical"}.get(name, name)
        scatter_clusters(
            petal,
            vis,
            f"{name}（簇编号已对齐真实类，便于对照）",
            OUTPUT_DIR / f"iris_{fig_name}.png",
            y_true=y,
            xlabel=names[2],
            ylabel=names[3],
        )

    table = pd.DataFrame(rows)[
        ["algorithm", "n_clusters", "n_noise", "rand_index", "ari", "silhouette"]
    ]
    out = ROOT / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "iris_four_algorithms.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    print("预处理：MinMaxScaler（不用 z-score，避免萼片宽度冲淡花瓣特征）。")
    print("默认参数：K-Means k=3；DBSCAN eps=0.13, min_samples=5；DPC t0=0.08。")
    print("作图颜色已按匈牙利算法对齐真实类；DBSCAN 无法切开重叠的 versicolor/virginica，常并成一类。")


if __name__ == "__main__":
    main()
