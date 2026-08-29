"""
密度峰值聚类（DPC, Rodriguez & Laio, 2014）。

t0：截断距离 dc 对应的邻域比例。
将全部成对距离升序排列后，取分位数 t0 作为 dc，
使得“圆（半径 dc）内样本数占数据集总样本数的比例”约为 t0。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances


class DensityPeakClustering:
    def __init__(self, t0: float = 0.02, n_clusters: int | None = 3):
        if not 0 < t0 < 1:
            raise ValueError("t0 必须在 (0, 1) 内")
        self.t0 = t0
        self.n_clusters = n_clusters
        self.dc_ = None
        self.rho_ = None
        self.delta_ = None
        self.gamma_ = None
        self.centers_ = None
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        dist = pairwise_distances(X)
        np.fill_diagonal(dist, 0.0)

        upper = dist[np.triu_indices(n, k=1)]
        self.dc_ = float(np.quantile(upper, self.t0))
        if self.dc_ <= 0:
            self.dc_ = float(np.min(upper[upper > 0]))

        # 高斯核局部密度（Rodriguez & Laio）；dc 由 t0 分位数确定
        self.rho_ = np.exp(-((dist / self.dc_) ** 2)).sum(axis=1)

        order = np.argsort(-self.rho_)
        self.delta_ = np.zeros(n)
        nearest_higher = np.full(n, -1, dtype=int)
        for rank, i in enumerate(order):
            if rank == 0:
                self.delta_[i] = dist[i].max()
                continue
            higher = order[:rank]
            j = higher[np.argmin(dist[i, higher])]
            nearest_higher[i] = j
            self.delta_[i] = dist[i, j]

        self.gamma_ = self.rho_ * self.delta_
        k = self.n_clusters
        if k is None:
            k = self._estimate_n_clusters()
        k = max(1, min(int(k), n))
        self.centers_ = np.argsort(-self.gamma_)[:k]

        labels = np.full(n, -1, dtype=int)
        for c, center in enumerate(self.centers_):
            labels[center] = c
        for i in order:
            if labels[i] == -1:
                parent = nearest_higher[i]
                labels[i] = labels[parent] if parent >= 0 else 0
        self.labels_ = labels
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def _estimate_n_clusters(self, max_k=8):
        g = np.sort(self.gamma_)[::-1]
        if len(g) < 3:
            return 2
        gaps = g[:-1] - g[1:]
        k = int(np.argmax(gaps[:max_k]) + 1)
        return max(2, k)
