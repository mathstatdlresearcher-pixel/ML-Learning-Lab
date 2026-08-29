"""1. 鸢尾花探索性分析与预处理。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data import iris_dataframe_unscaled, load_iris_data
from src.plotting import OUTPUT_DIR, setup_style


def main():
    setup_style()
    df = iris_dataframe_unscaled()
    numeric = df.select_dtypes(include="number").drop(columns=["target"])
    out_csv = ROOT / "outputs" / "tables"
    out_csv.mkdir(parents=True, exist_ok=True)

    desc = numeric.describe().T
    desc["missing"] = numeric.isna().sum()
    desc.to_csv(out_csv / "iris_describe.csv", encoding="utf-8-sig")

    corr = numeric.corr()
    corr.to_csv(out_csv / "iris_corr.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", ax=ax)
    ax.set_title("鸢尾花特征相关系数")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "iris_corr_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    pair = sns.pairplot(df, hue="species", vars=list(numeric.columns), corner=True)
    pair.fig.suptitle("鸢尾花成对特征分布", y=1.02)
    pair.savefig(OUTPUT_DIR / "iris_pairplot.png", bbox_inches="tight")
    plt.close(pair.fig)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    for ax, col in zip(axes, numeric.columns):
        sns.boxplot(data=df, x="species", y=col, ax=ax)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("各特征按类别的箱线图")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "iris_boxplots.png", bbox_inches="tight")
    plt.close(fig)

    X_raw, y, names, _, _ = load_iris_data(scale=False)
    X_std, _, _, _, _ = load_iris_data(scale="standard")
    X_mm, _, _, _, _ = load_iris_data(scale="minmax")
    pd.DataFrame(
        {
            "feature": names,
            "raw_mean": X_raw.mean(axis=0),
            "raw_std": X_raw.std(axis=0),
            "zscore_mean": X_std.mean(axis=0),
            "zscore_std": X_std.std(axis=0),
            "minmax_mean": X_mm.mean(axis=0),
            "minmax_std": X_mm.std(axis=0),
        }
    ).to_csv(out_csv / "iris_scale_compare.csv", index=False, encoding="utf-8-sig")
    print("预处理：z-score 会把萼片宽度提到与花瓣同等权重，三类更难分开；聚类实验采用 MinMax。")

    print("探索性分析完成。")
    print(f"- 样本数 {len(df)}，特征 {list(numeric.columns)}")
    print(f"- 类别分布:\n{df['species'].value_counts()}")
    print(f"- 缺失值: {int(numeric.isna().sum().sum())}")
    print(f"- 图表已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
