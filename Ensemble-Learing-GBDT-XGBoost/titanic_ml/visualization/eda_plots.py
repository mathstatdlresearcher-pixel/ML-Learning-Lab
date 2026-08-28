from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from titanic_ml.data.features import add_raw_features
from titanic_ml.visualization.heatmaps import annotated_heatmap
from titanic_ml.visualization.style import savefig


def plot_eda(raw: pd.DataFrame):
    summary = {
        "n_rows": int(len(raw)),
        "n_cols": int(raw.shape[1]),
        "survival_rate": float(raw["Survived"].mean()),
        "missing": raw.isnull().mean().sort_values(ascending=False).to_dict(),
    }
    plot_df = raw.copy()
    plot_df["SurvivedLabel"] = plot_df["Survived"].map({0: "未生存", 1: "生存"})

    fig, ax = plt.subplots(figsize=(8, 4.5))
    miss = raw.isnull().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    sns.barplot(x=miss.values, y=miss.index, ax=ax, color="#3b6ea5")
    ax.axvline(0.5, color="#c0392b", ls="--", label="50% 删除阈值")
    ax.set_xlabel("缺失比例")
    ax.set_title("各字段缺失比例")
    ax.legend()
    savefig("01_missing_ratio.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    sns.countplot(data=plot_df, x="SurvivedLabel", ax=axes[0], palette="Set2")
    axes[0].set_title("生存标签分布")
    sns.countplot(data=plot_df, x="Sex", hue="SurvivedLabel", ax=axes[1], palette="Set2")
    axes[1].set_title("性别与生存")
    sns.countplot(data=plot_df, x="Pclass", hue="SurvivedLabel", ax=axes[2], palette="Set2")
    axes[2].set_title("舱位等级与生存")
    savefig("02_survival_by_cat.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(data=plot_df, x="Age", hue="SurvivedLabel", bins=30, kde=True, ax=axes[0], palette="Set1")
    axes[0].set_title("年龄分布（按生存）")
    sns.histplot(data=plot_df, x="Fare", hue="SurvivedLabel", bins=30, kde=True, ax=axes[1], palette="Set1")
    axes[1].set_title("票价分布（按生存）")
    axes[1].set_xlim(0, 200)
    savefig("03_age_fare_dist.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=plot_df, x="Pclass", y="Age", hue="SurvivedLabel", ax=axes[0])
    axes[0].set_title("舱位-年龄箱线图")
    sns.barplot(data=raw, x="Embarked", y="Survived", ax=axes[1], ci=95, palette="muted")
    axes[1].set_title("登船港口平均生存率")
    savefig("04_box_embarked.png")

    tmp = add_raw_features(raw)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    order = tmp.groupby("Title")["Survived"].mean().sort_values(ascending=False).index
    sns.barplot(data=tmp, x="Title", y="Survived", order=order, ax=axes[0], palette="coolwarm")
    axes[0].set_title("称谓 Title 平均生存率")
    sns.barplot(data=tmp, x="FamilySize", y="Survived", ax=axes[1], palette="Blues_d")
    axes[1].set_title("家庭规模与生存率")
    savefig("05_title_family.png")

    fig, ax = plt.subplots(figsize=(8, 6.2))
    num = raw[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].corr()
    annotated_heatmap(num, ax, fmt=".2f", cmap="RdBu_r", center=0)
    ax.set_title("数值特征相关矩阵")
    savefig("06_corr_heatmap.png")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ct = pd.crosstab(raw["Sex"], raw["Pclass"], values=raw["Survived"], aggfunc="mean")
    annotated_heatmap(ct, ax, fmt=".2f", cmap="YlGnBu")
    ax.set_title("性别 × 舱位 生存率")
    savefig("07_sex_pclass_heatmap.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.violinplot(data=plot_df, x="SurvivedLabel", y="Age", ax=axes[0], palette="Set2")
    axes[0].set_title("年龄小提琴图（按生存）")
    sns.boxplot(data=plot_df, x="SurvivedLabel", y="Fare", ax=axes[1], palette="Set2", showfliers=False)
    axes[1].set_title("票价箱线图（隐藏极端值）")
    savefig("16_violin_box.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    stacked = pd.crosstab([raw["Pclass"], raw["Sex"]], raw["Survived"], normalize="index")
    stacked.plot(kind="bar", stacked=True, ax=ax, color=["#c0392b", "#27ae60"])
    ax.set_ylabel("比例")
    ax.set_title("舱位×性别 生存构成（堆叠）")
    ax.legend(["未生存", "生存"])
    savefig("17_stacked_pclass_sex.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=tmp, x="FamilyBin", y="Survived", order=["Alone", "Small", "Large"], ax=ax, palette="mako")
    ax.set_title("家庭类型分箱后的生存率")
    savefig("18_family_bin.png")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    deck_order = sorted(tmp["CabinDeck"].unique())
    sns.barplot(data=tmp, x="CabinDeck", y="Survived", order=deck_order, ax=ax, palette="crest")
    ax.set_title("舱位甲板（Cabin 首字母，U=未知）生存率")
    savefig("19_cabin_deck.png")
    return summary, tmp
