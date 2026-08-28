from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold, learning_curve

from titanic_ml.config.settings import MODEL_ORDER, RANDOM_STATE
from titanic_ml.visualization.heatmaps import annotated_heatmap
from titanic_ml.visualization.style import annotate_bars, axes_grid, savefig


def plot_split_comparison(results: dict, split: str, prefix: str, title_zh: str):
    names = [n for n in MODEL_ORDER if n in results]
    metric_names = ["Precision", "Recall", "F1", "AUC", "Accuracy"]
    data = pd.DataFrame({m: [results[n][split][m] for n in names] for m in metric_names}, index=names)

    fig, ax = plt.subplots(figsize=(12, 5.2))
    data.plot(kind="bar", ax=ax, colormap="tab10")
    ax.set_ylim(0.45, 1.08)
    ax.set_ylabel("分数")
    ax.set_title(f"{title_zh}指标对比")
    ax.legend(loc="lower right")
    ax.set_xticklabels(names, rotation=20, ha="right")
    annotate_bars(ax, fmt="{:.3f}", fontsize=6)
    savefig(f"{prefix}_metrics_bar.png")

    fig, ax = plt.subplots(figsize=(13, 6.6))
    heat = data.T.copy()
    heat.index.name = "指标"
    annotated_heatmap(heat, ax, fmt=".3f", cmap="YlGnBu", vmin=0.65, vmax=1.0, annot_size=11, x_rotation=0)
    ax.set_xlabel("模型")
    ax.set_ylabel("指标")
    ax.set_title(f"{title_zh}指标热力图")
    savefig(f"{prefix}_metrics_heatmap.png")

    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    y_true = results[names[0]][f"y_{split}"]
    for name in names:
        fpr, tpr, _ = roc_curve(y_true, results[name][f"proba_{split}"])
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={results[name][split]['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title_zh} ROC")
    ax.legend(loc="lower right", fontsize=8)
    savefig(f"{prefix}_roc.png")

    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    for name in names:
        p, r, _ = precision_recall_curve(y_true, results[name][f"proba_{split}"])
        ax.plot(r, p, lw=2, label=f"{name} (AP={results[name][split]['AP']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title_zh} Precision-Recall")
    ax.legend(loc="lower left", fontsize=8)
    savefig(f"{prefix}_pr.png")

    fig, axes = axes_grid(len(names), figsize=(13, 8.5))
    for ax, name in zip(axes, names):
        cm = pd.DataFrame(
            confusion_matrix(y_true, results[name][f"pred_{split}"]),
            index=["真实-未生存", "真实-生存"],
            columns=["预测-未生存", "预测-生存"],
        )
        annotated_heatmap(cm, ax, fmt="d", cmap="Blues", cbar=False, annot_size=12)
        ax.set_title(name)
    fig.suptitle(f"{title_zh}混淆矩阵")
    savefig(f"{prefix}_confusion.png")

    labels = metric_names
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.8, 6.8), subplot_kw=dict(polar=True))
    for name in names:
        vals = [data.loc[name, m] for m in labels] + [data.loc[name, labels[0]]]
        ax.plot(angles, vals, lw=2, label=name)
        ax.fill(angles, vals, alpha=0.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.55, 1.0)
    ax.set_title(f"{title_zh}五指标雷达图")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8)
    savefig(f"{prefix}_radar.png")
    return data


def plot_extra_model_viz(results, feature_names, importances, perm_imp):
    names = [n for n in MODEL_ORDER if n in results]

    fig, axes = axes_grid(len(names), figsize=(14, 9))
    for ax, name in zip(axes, names):
        imp = importances.get(name)
        if imp is None:
            ax.set_visible(False)
            continue
        s = pd.Series(imp, index=feature_names).sort_values(ascending=False).head(10)
        sns.barplot(x=s.values, y=s.index, ax=ax, color="#2c7fb8")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(f"{name} 内置重要性 Top10")
    fig.suptitle("特征重要性（树模型=impurity，逻辑回归=|系数|）")
    savefig("12_feature_importance.png")

    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = np.arange(len(names))
    w = 0.25
    ax.bar(x - w, [results[n]["train"]["AUC"] for n in names], w, label="训练集 AUC")
    ax.bar(x, [results[n]["val"]["AUC"] for n in names], w, label="验证集 AUC")
    ax.bar(x + w, [results[n]["test"]["AUC"] for n in names], w, label="测试集 AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.65, 1.08)
    ax.set_ylabel("AUC")
    ax.set_title("训练 / 验证 / 测试 AUC（过拟合诊断）")
    ax.legend()
    annotate_bars(ax, fmt="{:.3f}", fontsize=6)
    savefig("21_train_val_test_auc.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    y_true = results[names[0]]["y_test"]
    for name in names:
        frac_pos, mean_pred = calibration_curve(
            y_true, results[name]["proba_test"], n_bins=7, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("预测生存概率均值")
    ax.set_ylabel("实际正类比例")
    ax.set_title("测试集校准曲线")
    ax.legend(fontsize=8)
    savefig("22_calibration_test.png")

    preds = pd.DataFrame({n: results[n]["pred_test"] for n in names})
    agree = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            agree.loc[a, b] = float((preds[a] == preds[b]).mean())
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    annotated_heatmap(agree.astype(float), ax, fmt=".3f", cmap="Purples", vmin=0.65, vmax=1)
    ax.set_title("测试集预测一致率")
    savefig("23_pred_agreement_test.png")

    fig, axes = axes_grid(len(names), figsize=(14, 9))
    for ax, name in zip(axes, names):
        s = perm_imp.get(name)
        if s is None:
            continue
        top = s.head(8)
        sns.barplot(x=top.values, y=top.index, ax=ax, color="#6a51a3")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(f"{name} 置换重要性")
    fig.suptitle("测试集置换重要性（打乱后 AUC 下降）")
    savefig("24_permutation_importance.png")


def plot_learning_curves(results, X_train, y_train):
    names = [n for n in MODEL_ORDER if n in results]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    fig, axes = axes_grid(len(names), figsize=(14, 8))
    for ax, name in zip(axes, names):
        est = results[name]["best_estimator"]
        sizes, tr, te = learning_curve(
            est, X_train, y_train, cv=cv, scoring="roc_auc",
            train_sizes=np.linspace(0.35, 1.0, 4), n_jobs=-1,
            shuffle=True, random_state=RANDOM_STATE,
        )
        ax.plot(sizes, tr.mean(axis=1), marker="o", label="训练 AUC")
        ax.plot(sizes, te.mean(axis=1), marker="s", label="CV AUC")
        ax.fill_between(
            sizes, te.mean(axis=1) - te.std(axis=1), te.mean(axis=1) + te.std(axis=1), alpha=0.15
        )
        ax.set_title(name)
        ax.set_xlabel("训练样本数")
        ax.set_ylabel("ROC-AUC")
        ax.legend(fontsize=8)
        ax.set_ylim(0.65, 1.02)
    fig.suptitle("学习曲线（3 折 AUC）")
    savefig("25_learning_curves.png")


def plot_cv_val_test_auc(results):
    names = [n for n in MODEL_ORDER if n in results]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = np.arange(len(names))
    w = 0.25
    ax.bar(x - w, [results[n]["cv_best_auc"] for n in names], w, label="5折 CV AUC")
    ax.bar(x, [results[n]["val"]["AUC"] for n in names], w, label="验证集 AUC")
    ax.bar(x + w, [results[n]["test"]["AUC"] for n in names], w, label="测试集 AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.7, 1.05)
    ax.set_ylabel("AUC")
    ax.set_title("交叉验证 vs 验证集 vs 测试集 AUC")
    ax.legend()
    annotate_bars(ax, fmt="{:.3f}", fontsize=6)
    savefig("15_cv_val_test_auc.png")


def plot_cv_multi_heatmap(cv_multi_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6.2))
    heat = cv_multi_df.copy()
    heat.columns = [str(c) for c in heat.columns]
    annotated_heatmap(heat, ax, fmt=".3f", cmap="YlOrBr", annot_size=11, x_rotation=0)
    ax.set_xlabel("指标")
    ax.set_ylabel("模型")
    ax.set_title("最优参数在 5 折 CV 上的多指标均值")
    savefig("27_cv_multi_metrics.png")
