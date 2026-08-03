"""EDA 可视化。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from common.config import FIG_DIR, IRIS_CLASSES, IRIS_FEATURES, RESULT_DIR
from common.utils import save_fig, setup_font
from data_loader.prepare_data import load_abalone_data, load_blobs_data, load_iris_data


def eda_iris():
    data = load_iris_data()
    df = data["full"]
    out = FIG_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    df["class"].value_counts().reindex(IRIS_CLASSES).plot(
        kind="bar", color=["#4C78A8", "#F58518", "#54A24B"], ax=ax
    )
    ax.set_title("鸢尾花类别分布")
    ax.set_xlabel("类别")
    ax.set_ylabel("样本数")
    save_fig(fig, out / "iris_class_distribution.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    colors = ["#E45756", "#4C78A8", "#54A24B"]
    for i, feat in enumerate(IRIS_FEATURES):
        ax = axes[i // 2][i % 2]
        groups = [df.loc[df["class"] == c, feat] for c in IRIS_CLASSES]
        ax.hist(groups, color=colors, label=IRIS_CLASSES, bins=12)
        ax.set_title(feat)
        ax.legend(fontsize=8)
    fig.suptitle("鸢尾花特征分布")
    fig.tight_layout()
    save_fig(fig, out / "iris_feature_hist.png")

    g = sns.pairplot(df[IRIS_FEATURES + ["class"]], hue="class", corner=True)
    g.fig.suptitle("鸢尾花特征两两关系", y=1.02)
    save_fig(g.fig, out / "iris_pairplot.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df[IRIS_FEATURES].corr(), annot=True, cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("鸢尾花相关矩阵")
    save_fig(fig, out / "iris_corr.png")

    summary = df[IRIS_FEATURES + ["class_id"]].describe().T
    summary.to_csv(RESULT_DIR / "eda_iris_summary.csv")
    return summary


def eda_abalone():
    data = load_abalone_data(scale=False)
    full = data["full"]
    out = FIG_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(full["age"], bins=30, color="#4C78A8", edgecolor="white")
    ax.set_title("鲍鱼年龄分布")
    ax.set_xlabel("age")
    save_fig(fig, out / "abalone_age_hist.png")

    continuous = [c for c in full.columns if c != "age" and not str(c).startswith("Sex_")]
    fig, ax = plt.subplots(figsize=(10, 5))
    full[continuous].boxplot(ax=ax, rot=30)
    ax.set_title("鲍鱼连续特征箱线图")
    save_fig(fig, out / "abalone_feature_boxplot.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    corr = full[continuous + ["age"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("鲍鱼相关矩阵")
    save_fig(fig, out / "abalone_corr.png")

    top = corr["age"].drop("age").abs().sort_values(ascending=False).head(4).index
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, feat in enumerate(top):
        ax = axes[i // 2][i % 2]
        ax.scatter(full[feat], full["age"], alpha=0.35, s=12, color="#F58518")
        ax.set_xlabel(feat)
        ax.set_ylabel("age")
    fig.suptitle("关键特征 vs age")
    fig.tight_layout()
    save_fig(fig, out / "abalone_scatter_top.png")

    y_train, y_test = data["y_train"], data["y_test"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_train, bins=25, alpha=0.6, label="train", color="#4C78A8")
    ax.hist(y_test, bins=25, alpha=0.6, label="test", color="#E45756")
    ax.legend()
    ax.set_title("训练/测试年龄分布")
    save_fig(fig, out / "abalone_train_test_age.png")

    summary = full.describe().T
    summary.to_csv(RESULT_DIR / "eda_abalone_summary.csv")
    return summary


def eda_blobs():
    data = load_blobs_data()
    X, y = data["X"], data["y"]
    out = FIG_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=40)
    ax.set_title("线性二分类模拟数据")
    fig.colorbar(sc, ax=ax)
    save_fig(fig, out / "blobs_scatter.png")


def run_eda():
    setup_font()
    print(">>> EDA Iris")
    print(eda_iris())
    print(">>> EDA Abalone")
    print(eda_abalone())
    print(">>> EDA Blobs")
    eda_blobs()
    print(f"EDA 完成 -> {FIG_DIR / 'eda'}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    run_eda()