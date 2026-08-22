"""二.1 Breast Cancer 探索性分析。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import FIG_DIR, RESULT_DIR, cn
from common.plots import corr_heatmap, hist_grid
from common.style import COLORS, apply_theme, save_fig
from common.utils import save_csv, save_json
from data_loader.prepare_data import load_breast_raw


def run() -> dict:
    apply_theme()
    out = FIG_DIR / "01_breast_eda"
    out.mkdir(parents=True, exist_ok=True)

    df = load_breast_raw()
    feat_cols = [c for c in df.columns if c not in {"diagnosis", "diagnosis_raw"}]
    X = df[feat_cols]
    y = df["diagnosis"]

    desc = X.describe().T
    desc["missing"] = X.isna().sum()
    desc["skew"] = X.skew()
    save_csv(desc.reset_index().rename(columns={"index": "feature"}), RESULT_DIR / "breast_eda_describe.csv")

    cls = pd.DataFrame(
        {
            "label": ["良性(0)", "恶性(1)"],
            "count": [(y == 0).sum(), (y == 1).sum()],
        }
    )
    cls["ratio"] = cls["count"] / cls["count"].sum()
    save_csv(cls, RESULT_DIR / "breast_eda_class_balance.csv")

    info = {
        "n_samples": int(len(df)),
        "n_features": len(feat_cols),
        "n_missing": int(df.isna().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "malignant": int((y == 1).sum()),
        "benign": int((y == 0).sum()),
    }
    save_json(info, RESULT_DIR / "breast_eda_summary.json")

    # 类别分布
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    bars = ax.bar(cls["label"], cls["count"], color=[COLORS["teal"], COLORS["coral"]], edgecolor="white")
    ax.set_ylabel("样本数")
    ax.set_title("Breast Cancer 类别分布", fontweight="bold")
    for b, r in zip(bars, cls["ratio"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{r:.1%}", ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out / "class_balance.png")

    hist_grid(df, feat_cols, "Breast Cancer 特征分布总览", out / "distributions.png")

    # 按类别对比若干关键特征
    corr_y = X.corrwith(y).abs().sort_values(ascending=False)
    top8 = corr_y.head(8).index.tolist()
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.0))
    for ax, col in zip(axes.ravel(), top8):
        sns.boxplot(
            data=df,
            x="diagnosis",
            y=col,
            hue="diagnosis",
            palette={0: COLORS["teal"], 1: COLORS["coral"]},
            legend=False,
            ax=ax,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["良性", "恶性"])
        ax.set_xlabel("")
        ax.set_ylabel(cn(col))
        ax.set_title(cn(col), fontsize=11, fontweight="bold")
    fig.suptitle("与诊断相关性最高的 8 个特征 · 箱线图", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out / "top8_box_by_class.png")

    # 相关热力图（全量不标数字，子集标注）
    corr_heatmap(X, "Breast Cancer 特征相关热力图", out / "corr_heatmap.png", annot=False)
    top12 = corr_y.head(12).index.tolist()
    corr_heatmap(X[top12], "与诊断最相关的 12 个特征热力图", out / "corr_heatmap_top12.png", annot=True)

    # 与诊断相关度
    fig, ax = plt.subplots(figsize=(8.8, 8.2))
    vals = X.corrwith(y).sort_values()
    colors = [COLORS["coral"] if v < 0 else COLORS["teal"] for v in vals]
    ax.barh([cn(i) for i in vals.index], vals.values, color=colors, edgecolor="white")
    ax.axvline(0, color=COLORS["slate"], lw=1.2)
    ax.set_xlabel("与恶性标签的相关系数")
    ax.set_title("各特征与乳腺癌诊断的相关度", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out / "corr_with_label.png")
    save_csv(
        pd.DataFrame({"feature": corr_y.index, "abs_corr": corr_y.values, "cn": [cn(i) for i in corr_y.index]}),
        RESULT_DIR / "breast_eda_corr_with_label.csv",
    )

    # 前 5 个特征散点矩阵
    top5 = corr_y.head(5).index.tolist()
    plot_df = df[top5 + ["diagnosis"]].copy()
    plot_df["类别"] = plot_df["diagnosis"].map({0: "良性", 1: "恶性"})
    g = sns.pairplot(
        plot_df,
        vars=top5,
        hue="类别",
        palette={"良性": COLORS["teal"], "恶性": COLORS["coral"]},
        corner=True,
        plot_kws={"s": 22, "alpha": 0.7, "edgecolor": "white", "linewidth": 0.2},
    )
    g.fig.suptitle("关键特征散点矩阵（按诊断着色）", y=1.02, fontsize=15, fontweight="bold")
    save_fig(g.fig, out / "pairplot_top5.png", close=False)
    plt.close(g.fig)

    print("  Breast EDA 完成")
    return info


if __name__ == "__main__":
    run()
