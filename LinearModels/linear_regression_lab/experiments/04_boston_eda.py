"""
实验脚本 04：Boston 房价探索性分析与预处理
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FIG_DIR, REPORT_DIR
from data.boston import load_boston, preprocess_boston
from utils.style import COLORS, apply_theme, save_fig
from utils.viz import plot_corr_heatmap


def run() -> dict:
    apply_theme()
    out_dir = FIG_DIR / "04_boston_eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_boston()
    df.to_csv(REPORT_DIR / "boston_raw.csv", index=False, encoding="utf-8-sig")

    # 缺失与描述
    missing = df.isna().sum()
    desc = df.describe().T
    desc.to_csv(REPORT_DIR / "boston_describe.csv", encoding="utf-8-sig")

    # 分布总览
    num_cols = df.columns.tolist()
    n = len(num_cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.1 * nrows))
    axes = axes.ravel()
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col], bins=24, color=COLORS["teal"], edgecolor="white", alpha=0.88)
        axes[i].set_title(col, fontsize=10, fontweight="bold")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Boston 房价 · 特征分布总览", fontsize=15, fontweight="bold", color=COLORS["ink"])
    fig.tight_layout()
    save_fig(fig, out_dir / "distributions.png")

    # 相关热力图
    plot_corr_heatmap(df, title="Boston 特征相关热力图", save_path=out_dir / "corr_heatmap.png")

    # 与房价相关度条形图
    corr = df.corr(numeric_only=True)["MEDV"].drop("MEDV").sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [COLORS["coral"] if v < 0 else COLORS["teal"] for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors, edgecolor="white")
    ax.axvline(0, color=COLORS["slate"], lw=1.2)
    ax.set_xlabel("与 MEDV 的相关系数")
    ax.set_title("各特征与房价相关度", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir / "corr_with_price.png")

    # 关键特征散点
    top_feats = corr.abs().sort_values(ascending=False).head(3).index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    for ax, feat, color in zip(axes, top_feats, COLORS["series"]):
        ax.scatter(df[feat], df["MEDV"], s=28, alpha=0.65, c=color, edgecolors="white", linewidths=0.3)
        # 简单趋势线
        z = np.polyfit(df[feat], df["MEDV"], 1)
        xs = np.linspace(df[feat].min(), df[feat].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), color=COLORS["slate"], ls="--", lw=1.5)
        ax.set_xlabel(feat)
        ax.set_ylabel("MEDV")
        ax.set_title(f"{feat} vs MEDV", fontweight="bold")
    fig.suptitle("关键相关特征散点与趋势", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir / "key_scatter.png")

    # 预处理
    X, y, info = preprocess_boston(df)
    clean = X.copy()
    clean["MEDV"] = y.values
    clean.to_csv(REPORT_DIR / "boston_processed.csv", index=False, encoding="utf-8-sig")

    # 箱线图：处理前后房价
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(y=df["MEDV"], ax=axes[0], color=COLORS["cyan"])
    axes[0].set_title("原始 MEDV", fontweight="bold")
    sns.boxplot(y=y, ax=axes[1], color=COLORS["teal"])
    axes[1].set_title("去除封顶值后 MEDV", fontweight="bold")
    fig.suptitle("房价分布：异常/封顶处理对比", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir / "price_boxplot.png")

    summary = {
        "shape_raw": list(df.shape),
        "missing": missing.to_dict(),
        "preprocess": {
            "n_after": int(len(y)),
            "dropped_capped": info["dropped_capped"],
            "selected_features": info["selected_features"],
        },
        "corr_with_medv": corr.to_dict(),
        "figures": sorted(str(p) for p in out_dir.glob("*.png")),
    }
    with open(REPORT_DIR / "04_boston_eda.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("【04】Boston EDA 完成")
    print(f"原始样本: {df.shape}, 处理后: {clean.shape}")
    print(f"入选特征: {info['selected_features']}")
    print(f"图片目录: {out_dir}")
    return summary


if __name__ == "__main__":
    run()
