"""
探究 K-Means 中 k 对鸢尾花聚类性能的影响，并用肘部法、轮廓系数选 k。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import runtime  # noqa: F401

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.data import load_iris_data
from src.metrics_util import evaluate_clustering
from src.plotting import OUTPUT_DIR, line_plot, setup_style


def main():
    setup_style()
    X, y, _, _, _ = load_iris_data(scale="minmax")
    ks = list(range(2, 11))
    inertias, sils, ris, aris = [], [], [], []

    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        pred = km.fit_predict(X)
        m = evaluate_clustering(X, y, pred)
        inertias.append(km.inertia_)
        sils.append(m["silhouette"])
        ris.append(m["rand_index"])
        aris.append(m["ari"])

    table = pd.DataFrame(
        {
            "k": ks,
            "inertia": inertias,
            "silhouette": sils,
            "rand_index": ris,
            "ari": aris,
        }
    )
    out = ROOT / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "kmeans_k_sweep.csv", index=False, encoding="utf-8-sig")

    line_plot(
        ks,
        {"SSE/inertia": inertias},
        "k",
        "簇内平方和 SSE",
        "肘部法：K-Means 的 k 与 SSE",
        OUTPUT_DIR / "kmeans_elbow.png",
    )
    line_plot(
        ks,
        {"轮廓系数": sils, "兰德系数 RI": ris, "ARI": aris},
        "k",
        "指标值",
        "k 对轮廓系数与兰德系数的影响",
        OUTPUT_DIR / "kmeans_k_metrics.png",
    )

    best_sil_k = ks[int(np.nanargmax(sils))]
    best_ari_k = ks[int(np.nanargmax(aris))]
    # 肘部：SSE 二阶差分最大处
    sse = np.asarray(inertias)
    second = sse[:-2] - 2 * sse[1:-1] + sse[2:]
    elbow_k = ks[int(np.argmax(second)) + 1]

    print(table.to_string(index=False))
    print(f"轮廓系数最大的 k = {best_sil_k}")
    print(f"相对真实标签 ARI 最大的 k = {best_ari_k}（仅作对照，实际无标签时不可用）")
    print(f"肘部法（二阶差分）建议 k = {elbow_k}")
    print("鸢尾花真实类别数为 3，与肘部/轮廓结果对照即可讨论选 k 方法。")


if __name__ == "__main__":
    main()
