"""三.1 Boston 房价探索性分析。"""

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
from data_loader.prepare_data import load_boston_raw


def run() -> dict:
    apply_theme()
    out = FIG_DIR / "05_boston_eda"
    out.mkdir(parents=True, exist_ok=True)

    df = load_boston_raw()
    feat_cols = [c for c in df.columns if c != "MEDV"]

    desc = df.describe().T
    desc["missing"] = df.isna().sum()
    desc["skew"] = df.skew(numeric_only=True)
    save_csv(desc.reset_index().rename(columns={"index": "feature"}), RESULT_DIR / "boston_eda_describe.csv")

    info = {
        "n_samples": int(len(df)),
        "n_features": len(feat_cols),
        "n_missing": int(df.isna().sum().sum()),
        "n_capped50": int((df["MEDV"] >= 50).sum()),
        "price_mean": float(df["MEDV"].mean()),
        "price_median": float(df["MEDV"].median()),
        "chas_rate": float(df["CHAS"].mean()) if "CHAS" in df.columns else None,
    }
    save_json(info, RESULT_DIR / "boston_eda_summary.json")

    hist_grid(df, feat_cols + ["MEDV"], "Boston 房价特征分布总览", out / "distributions.png")

    # 房价单独分布
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    axes[0].hist(df["MEDV"], bins=24, color=COLORS["teal"], edgecolor="white")
    axes[0].axvline(df["MEDV"].mean(), color=COLORS["coral"], ls="--", label=f"均值 {df['MEDV'].mean():.1f}")
    axes[0].axvline(50, color=COLORS["amber"], ls=":", label="封顶 50")
    axes[0].set_title("房价 MEDV 直方图", fontweight="bold")
    axes[0].legend(frameon=False)
    axes[1].boxplot(df["MEDV"], vert=True, patch_artist=True, boxprops=dict(facecolor=COLORS["teal"], alpha=0.7))
    axes[1].set_title("房价箱线图", fontweight="bold")
    axes[1].set_xticklabels(["MEDV"])
    fig.tight_layout()
    save_fig(fig, out / "price_distribution.png")

    corr_heatmap(df, "Boston 特征相关热力图", out / "corr_heatmap.png", annot=True, fmt=".2f")

    corr = df.corr(numeric_only=True)["MEDV"].drop("MEDV").sort_values()
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    colors = [COLORS["coral"] if v < 0 else COLORS["teal"] for v in corr.values]
    ax.barh([cn(i) for i in corr.index], corr.values, color=colors, edgecolor="white")
    ax.axvline(0, color=COLORS["slate"], lw=1.2)
    ax.set_xlabel("与房价 MEDV 的相关系数")
    ax.set_title("各特征与房价相关度", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out / "corr_with_price.png")
    save_csv(
        pd.DataFrame({"feature": corr.index, "corr": corr.values, "cn": [cn(i) for i in corr.index]}),
        RESULT_DIR / "boston_eda_corr_with_price.csv",
    )

    top3 = corr.abs().sort_values(ascending=False).head(3).index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    for ax, feat, color in zip(axes, top3, COLORS["series"]):
        ax.scatter(df[feat], df["MEDV"], s=26, alpha=0.7, c=color, edgecolors="white", linewidths=0.25)
        z = np.polyfit(df[feat], df["MEDV"], 1)
        xs = np.linspace(df[feat].min(), df[feat].max(), 80)
        ax.plot(xs, np.poly1d(z)(xs), color=COLORS["slate"], ls="--", lw=1.5)
        ax.set_xlabel(cn(feat))
        ax.set_ylabel("房价 MEDV")
        ax.set_title(f"{cn(feat)} vs 房价", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out / "top3_scatter.png")

    if "CHAS" in df.columns:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        data = df.copy()
        data["临河"] = data["CHAS"].map({0: "不临河", 1: "临河", 0.0: "不临河", 1.0: "临河"})
        sns.boxplot(
            data=data,
            x="临河",
            y="MEDV",
            hue="临河",
            palette={"不临河": COLORS["teal"], "临河": COLORS["coral"]},
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("房价 MEDV")
        ax.set_title("临河与否对房价的影响", fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out / "chas_boxplot.png")

    print("  Boston EDA 完成")
    return info


if __name__ == "__main__":
    run()
