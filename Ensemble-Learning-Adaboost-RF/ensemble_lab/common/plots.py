"""通用绘图：分布、热力图、指标柱、ROC、混淆矩阵、特征重要性。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve

from .config import cn
from .style import COLORS, save_fig

CMAP = LinearSegmentedColormap.from_list(
    "teal_coral",
    ["#0F766E", "#5EEAD4", "#F8FAFC", "#FDA4AF", "#E11D48"],
)


def hist_grid(df: pd.DataFrame, columns: Sequence[str], title: str, save_path: Path, bins: int = 24):
    cols = list(columns)
    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 3.05 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=bins, color=COLORS["teal"], edgecolor="white", alpha=0.9)
        axes[i].set_title(cn(col), fontsize=10, fontweight="bold")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=15, fontweight="bold", color=COLORS["ink"])
    fig.tight_layout()
    save_fig(fig, save_path)


def corr_heatmap(df: pd.DataFrame, title: str, save_path: Path, annot: bool = True, fmt: str = ".2f"):
    corr = df.corr(numeric_only=True)
    size = max(7.5, 0.38 * corr.shape[0] + 4.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    labels = [cn(c) for c in corr.columns]
    sns.heatmap(
        corr,
        mask=mask,
        cmap=CMAP,
        center=0,
        annot=annot,
        fmt=fmt,
        annot_kws={"size": 8 if corr.shape[0] <= 16 else 6},
        square=True,
        linewidths=0.4,
        linecolor="white",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    fig.tight_layout()
    save_fig(fig, save_path)


def metric_bars(
    results: Dict[str, Dict[str, float]],
    metrics: Sequence[str],
    title: str,
    save_path: Path,
):
    models = list(results.keys())
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.15 * len(metrics), 4.7))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        vals = [results[m].get(metric, np.nan) for m in models]
        bars = ax.bar(x, vals, width=0.62, color=COLORS["series"][: len(models)], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10, rotation=12)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        finite = [v for v in vals if np.isfinite(v)]
        if finite:
            lo, hi = min(finite), max(finite)
            pad = 0.08 * (hi - lo + 1e-6)
            ax.set_ylim(max(0, lo - pad) if metric != "AUC" else max(0.8, lo - 0.03), hi + pad)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, save_path)


def roc_overlay(y_true, scores: Dict[str, np.ndarray], title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    y_true = np.asarray(y_true).ravel()
    for i, (name, score) in enumerate(scores.items()):
        fpr, tpr, _ = roc_curve(y_true, np.asarray(score).ravel())
        ax.plot(fpr, tpr, color=COLORS["series"][i % len(COLORS["series"])], label=f"{name}  AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color=COLORS["slate"], lw=1.2)
    ax.set_xlabel("假阳性率 FPR")
    ax.set_ylabel("真正率 TPR")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    save_fig(fig, save_path)


def confusion_grid(y_true, preds: Dict[str, np.ndarray], labels: Sequence[str], title: str, save_path: Path):
    names = list(preds.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4.2))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            preds[name],
            display_labels=list(labels),
            cmap="YlGn",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.grid(False)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, save_path)


def importance_bars(names: Sequence[str], values: Sequence[float], title: str, save_path: Path, topk: int = 15):
    names = [cn(n) for n in names]
    values = np.asarray(values, dtype=float)
    order = np.argsort(np.abs(values))[::-1][:topk]
    names = [names[i] for i in order][::-1]
    values = values[order][::-1]
    colors = [COLORS["coral"] if v < 0 else COLORS["teal"] for v in values]
    fig, ax = plt.subplots(figsize=(8.6, max(4.0, 0.38 * len(names) + 1.6)))
    ax.barh(names, values, color=colors, edgecolor="white", height=0.72)
    ax.axvline(0, color=COLORS["slate"], lw=1.2)
    ax.set_xlabel("重要性 / 系数")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    fig.tight_layout()
    save_fig(fig, save_path)


def pred_vs_true(y_true, predictions: Dict[str, np.ndarray], title: str, save_path: Path):
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(4.35 * n, 4.35), squeeze=False)
    y_true = np.asarray(y_true).ravel()
    lo = min(y_true.min(), min(np.min(v) for v in predictions.values()))
    hi = max(y_true.max(), max(np.max(v) for v in predictions.values()))
    pad = 0.06 * (hi - lo + 1e-9)
    for ax, (name, y_pred) in zip(axes[0], predictions.items()):
        y_pred = np.asarray(y_pred).ravel()
        ax.scatter(y_true, y_pred, s=36, alpha=0.75, c=COLORS["teal"], edgecolors="white", linewidths=0.3)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=COLORS["slate"], ls="--", lw=1.6)
        ax.set_xlabel("真实值")
        ax.set_ylabel("预测值")
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, save_path)


def line_metrics(
    x,
    curves: Dict[str, Sequence[float]],
    xlabel: str,
    title: str,
    save_path: Path,
    x_as_str: bool = False,
):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    xs = list(x)
    for i, (name, ys) in enumerate(curves.items()):
        ax.plot(xs, list(ys), marker="o", markersize=5, color=COLORS["series"][i % len(COLORS["series"])], label=name)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.legend(frameon=False)
    if x_as_str:
        ax.set_xticks(xs)
        ax.set_xticklabels([str(v) for v in xs], rotation=20)
    fig.tight_layout()
    save_fig(fig, save_path)
