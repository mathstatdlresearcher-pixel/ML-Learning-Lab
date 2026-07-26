"""清晰可视化：无颜色说明图例 / 无 colorbar 注释。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from .style import COLORS, save_fig

CMAP = LinearSegmentedColormap.from_list(
    "teal_coral",
    ["#0F766E", "#5EEAD4", "#F8FAFC", "#FDA4AF", "#E11D48"],
)


def plot_corr_heatmap(df: pd.DataFrame, title: str, save_path: Optional[Path] = None):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=CMAP,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 11},
        square=True,
        linewidths=0.8,
        linecolor="white",
        cbar=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_vif_bars(vif_df: pd.DataFrame, title: str, save_path: Optional[Path] = None):
    df = vif_df.sort_values("VIF")
    vals = df["VIF"].clip(lower=1.0)
    fig, ax = plt.subplots(figsize=(8.5, max(3.8, 0.55 * len(df) + 1.5)))
    ax.barh(df["feature"], vals, color=COLORS["teal"], edgecolor="white", height=0.65)
    ax.set_xscale("log")
    ax.axvline(5, color=COLORS["amber"], ls="--", lw=1.6)
    ax.axvline(10, color=COLORS["coral"], ls="--", lw=1.6)
    ax.set_xlabel("VIF")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    for y, v in zip(df["feature"], df["VIF"]):
        ax.text(max(v, 1.05), y, f" {v:.1f}", va="center", fontsize=11)
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_3d_scatter(x1, x2, y, title: str, save_path: Optional[Path] = None):
    fig = plt.figure(figsize=(8.2, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x1, x2, y, c=COLORS["teal"], s=42, alpha=0.85, edgecolors="white", linewidths=0.25)
    ax.set_xlabel("x1", labelpad=8)
    ax.set_ylabel("x2", labelpad=8)
    ax.set_zlabel("y", labelpad=8)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_xy_scatter(x, y, title: str, xlabel: str, ylabel: str = "y", save_path: Optional[Path] = None):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(x, y, s=46, alpha=0.8, c=COLORS["teal"], edgecolors="white", linewidths=0.4)
    coef = np.polyfit(np.asarray(x), np.asarray(y), 1)
    xs = np.linspace(np.min(x), np.max(x), 100)
    ax.plot(xs, np.poly1d(coef)(xs), color=COLORS["coral"], lw=2.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_metric_bars(
    results: Dict[str, Dict[str, float]],
    metrics: Sequence[str] = ("R2", "RMSE", "MAE"),
    title: str = "模型对比",
    save_path: Optional[Path] = None,
):
    models = list(results.keys())
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 4.8))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        vals = [results[m][metric] for m in models]
        bars = ax.bar(x, vals, width=0.62, color=COLORS["teal"], edgecolor="white", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=12)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_title(metric, fontsize=15, fontweight="bold")
        ymin = min(vals)
        ymax = max(vals)
        # 拉开视觉：对接近的指标放大纵轴局部
        if metric == "R2":
            lo = max(0.0, ymin - 0.05)
            hi = min(1.05, ymax + 0.05)
            if hi - lo < 0.08:
                mid = (hi + lo) / 2
                lo, hi = mid - 0.06, mid + 0.06
            ax.set_ylim(lo, hi)
        span = abs(ymax - ymin)
        pad = 0.04 * (abs(ymax) + 1e-9) if span < 1e-6 else 0.08 * span
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + pad,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    fig.suptitle(title, fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_pred_vs_true(
    y_true,
    predictions: Dict[str, np.ndarray],
    title: str,
    save_path: Optional[Path] = None,
):
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.4), squeeze=False)
    y_true = np.asarray(y_true).ravel()
    lo = min(y_true.min(), min(np.min(v) for v in predictions.values()))
    hi = max(y_true.max(), max(np.max(v) for v in predictions.values()))
    pad = 0.06 * (hi - lo + 1e-9)

    for ax, (name, y_pred) in zip(axes[0], predictions.items()):
        y_pred = np.asarray(y_pred).ravel()
        ax.scatter(y_true, y_pred, s=40, alpha=0.75, c=COLORS["teal"], edgecolors="white", linewidths=0.35)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=COLORS["slate"], ls="--", lw=1.8)
        ax.set_xlabel("真实值")
        ax.set_ylabel("预测值")
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(title, fontsize=17, fontweight="bold", y=1.03)
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_series_compare(
    y_true,
    predictions: Dict[str, np.ndarray],
    title: str,
    save_path: Optional[Path] = None,
):
    """多模型对比用子图，避免颜色图例。"""
    y_true = np.asarray(y_true).ravel()
    order = np.argsort(y_true)
    y_sorted = y_true[order]
    idx = np.arange(len(y_sorted))
    names = list(predictions.keys())
    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(11.5, 2.4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        y_pred = np.asarray(predictions[name]).ravel()[order]
        ax.plot(idx, y_sorted, color=COLORS["slate"], lw=2.4, label="真实值")
        ax.plot(idx, y_pred, color=COLORS["teal"], lw=1.8, label=name)
        ax.set_ylabel(name, fontsize=12, fontweight="bold")
    axes[-1].set_xlabel("样本（按真实值排序）")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_series_one_vs_true(
    y_true,
    y_pred,
    model_name: str,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """单个模型预测曲线 vs 真实值（独立大图）。"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    order = np.argsort(y_true)
    y_sorted = y_true[order]
    y_hat = y_pred[order]
    idx = np.arange(len(y_sorted))

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.plot(idx, y_sorted, color=COLORS["slate"], lw=2.8, label="真实值")
    ax.plot(idx, y_hat, color=COLORS["teal"], lw=2.0, label=f"{model_name} 预测")
    ax.fill_between(idx, y_sorted, y_hat, color=COLORS["teal"], alpha=0.12)
    ax.set_xlabel("样本（按真实值排序）")
    ax.set_ylabel("目标值")
    ax.set_title(title or f"{model_name} vs 真实值", fontsize=16, fontweight="bold", pad=12)
    ax.legend(loc="best", fontsize=12, frameon=False)
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_residuals(y_true, y_pred, title: str, save_path: Optional[Path] = None):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    resid = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].scatter(y_pred, resid, s=40, alpha=0.75, c=COLORS["teal"], edgecolors="white", linewidths=0.35)
    axes[0].axhline(0, color=COLORS["slate"], ls="--", lw=1.6)
    axes[0].set_xlabel("预测值")
    axes[0].set_ylabel("残差")
    axes[0].set_title("残差 vs 预测", fontsize=14, fontweight="bold")

    axes[1].hist(resid, bins=20, color=COLORS["teal"], edgecolor="white", alpha=0.9)
    axes[1].axvline(0, color=COLORS["slate"], ls="--", lw=1.6)
    axes[1].set_xlabel("残差")
    axes[1].set_ylabel("频数")
    axes[1].set_title("残差分布", fontsize=14, fontweight="bold")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_coef_compare(
    feature_names: Sequence[str],
    true_coef: Dict[str, float],
    estimated: Dict[str, Sequence[float]],
    all_feature_names: Optional[Sequence[str]] = None,
    title: str = "系数对比",
    save_path: Optional[Path] = None,
):
    """只画指定特征（通常为相关特征），子图分模型，无颜色图例。"""
    names = list(feature_names)
    true_vals = np.array([true_coef.get(n, 0.0) for n in names], dtype=float)
    models = list(estimated.keys())
    full_names = list(all_feature_names) if all_feature_names is not None else names

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.8), sharey=True)
    if n == 1:
        axes = [axes]
    x = np.arange(len(names))
    for ax, name in zip(axes, models):
        coefs = np.asarray(estimated[name], dtype=float)
        if len(coefs) == len(full_names):
            idx = [full_names.index(f) for f in names]
            vals = coefs[idx]
        else:
            vals = coefs[: len(names)]
        ax.bar(x - 0.18, true_vals, width=0.36, color=COLORS["slate"], edgecolor="white")
        ax.bar(x + 0.18, vals, width=0.36, color=COLORS["teal"], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.axhline(0, color=COLORS["grid"], lw=1.0)
    axes[0].set_ylabel("系数")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_coef_stability(
    feature_names: Sequence[str],
    coef_runs: Dict[str, np.ndarray],
    feature_index: Optional[Sequence[int]] = None,
    title: str = "系数稳定性",
    save_path: Optional[Path] = None,
):
    models = list(coef_runs.keys())
    idx = list(feature_index) if feature_index is not None else list(range(len(feature_names)))
    names = list(feature_names)
    n_feat = len(names)
    fig, axes = plt.subplots(1, n_feat, figsize=(3.2 * n_feat, 4.8), sharey=False)
    if n_feat == 1:
        axes = [axes]

    for ax, feat, j in zip(axes, names, idx):
        data = [coef_runs[m][:, j] for m in models]
        bp = ax.boxplot(data, labels=models, patch_artist=True, widths=0.55)
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS["teal"])
            patch.set_alpha(0.75)
        ax.set_title(feat, fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
        ax.axhline(0, color=COLORS["grid"], lw=1.0)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_alpha_path(
    alphas: Iterable[float],
    scores: Sequence[float],
    best_alpha: float,
    title: str,
    ylabel: str = "CV MSE",
    save_path: Optional[Path] = None,
):
    alphas = np.asarray(list(alphas), dtype=float)
    scores = np.asarray(scores, dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(alphas, scores, color=COLORS["teal"], marker="o", markersize=5)
    ax.axvline(best_alpha, color=COLORS["coral"], ls="--", lw=2.0)
    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  (α={best_alpha:.3g})", fontsize=16, fontweight="bold", pad=12)
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_lars_stability(
    selection_matrix: np.ndarray,
    feature_names: Sequence[str],
    title: str,
    save_path: Optional[Path] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [2.1, 1]})
    sns.heatmap(
        selection_matrix,
        cmap=LinearSegmentedColormap.from_list("sel", ["#F1F5F9", "#0F766E"]),
        cbar=False,
        yticklabels=max(1, selection_matrix.shape[0] // 10),
        xticklabels=list(feature_names),
        ax=axes[0],
        linewidths=0.3,
        linecolor="white",
    )
    axes[0].set_xlabel("特征")
    axes[0].set_ylabel("重复次数")
    axes[0].set_title("入选矩阵", fontsize=14, fontweight="bold")

    freq = selection_matrix.mean(axis=0)
    axes[1].barh(list(feature_names), freq, color=COLORS["teal"], edgecolor="white", height=0.65)
    axes[1].set_xlim(0, 1.08)
    axes[1].set_xlabel("入选频率")
    axes[1].set_title("入选频率", fontsize=14, fontweight="bold")
    for i, v in enumerate(freq):
        axes[1].text(v + 0.03, i, f"{v:.0%}", va="center", fontsize=12, fontweight="bold")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_feature_importance(
    names: Sequence[str],
    coefs: Sequence[float],
    title: str,
    save_path: Optional[Path] = None,
):
    names = list(names)
    coefs = np.asarray(coefs, dtype=float)
    order = np.argsort(np.abs(coefs))
    names = [names[i] for i in order]
    coefs = coefs[order]

    fig, ax = plt.subplots(figsize=(8.2, max(3.8, 0.48 * len(names) + 1.6)))
    ax.barh(names, coefs, color=COLORS["teal"], edgecolor="white", height=0.7)
    ax.axvline(0, color=COLORS["slate"], lw=1.4)
    ax.set_xlabel("系数")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_pair_overview(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    title: str,
    save_path: Optional[Path] = None,
):
    cols = list(features) + [target]
    g = sns.pairplot(
        df[cols],
        corner=True,
        diag_kind="hist",
        plot_kws={"s": 26, "alpha": 0.65, "color": COLORS["teal"], "edgecolor": "white", "linewidth": 0.25},
        diag_kws={"color": COLORS["teal"], "edgecolor": "white"},
    )
    g.fig.suptitle(title, y=1.02, fontsize=16, fontweight="bold", color=COLORS["ink"])
    g.fig.patch.set_facecolor(COLORS["bg"])
    if save_path:
        save_fig(g.fig, save_path, close=False)
        plt.close(g.fig)
    return g
