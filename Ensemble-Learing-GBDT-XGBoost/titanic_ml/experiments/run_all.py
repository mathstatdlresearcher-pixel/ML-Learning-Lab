"""Full experiment: EDA, preprocess, grid search, plots, markdown report."""
from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from titanic_ml.config.settings import MODEL_ORDER, OUT_DIR, RANDOM_STATE
from titanic_ml.data.features import add_raw_features
from titanic_ml.data.loading import load_raw
from titanic_ml.data.preprocess import bin_and_dummy, drop_high_missing, fill_missing
from titanic_ml.models.importance import native_importance
from titanic_ml.models.metrics import jsonable_params, metrics_dict
from titanic_ml.models.zoo import build_model_zoo
from titanic_ml.reporting.markdown import write_markdown_report
from titanic_ml.visualization.eda_plots import plot_eda
from titanic_ml.visualization.model_plots import (
    plot_cv_multi_heatmap,
    plot_cv_val_test_auc,
    plot_extra_model_viz,
    plot_learning_curves,
    plot_split_comparison,
)
from titanic_ml.visualization.style import apply_plot_style, savefig

warnings.filterwarnings("ignore")


def main() -> None:
    apply_plot_style()
    train_raw, kaggle_raw = load_raw()
    eda_summary, feat_preview = plot_eda(train_raw)

    dev_part, labeled_test_part = train_test_split(
        train_raw, test_size=0.2, stratify=train_raw["Survived"], random_state=RANDOM_STATE
    )
    train_part, val_part = train_test_split(
        dev_part, test_size=0.2, stratify=dev_part["Survived"], random_state=RANDOM_STATE
    )
    train_part = train_part.reset_index(drop=True)
    val_part = val_part.reset_index(drop=True)
    labeled_test_part = labeled_test_part.reset_index(drop=True)
    kaggle_part = kaggle_raw.copy().reset_index(drop=True)

    ticket_freq = train_part["Ticket"].value_counts()
    train_f = add_raw_features(train_part, ticket_freq)
    val_f = add_raw_features(val_part, ticket_freq)
    lab_f = add_raw_features(labeled_test_part, ticket_freq)
    kag_f = add_raw_features(kaggle_part, ticket_freq)

    drop_cols, miss_ratio, train_f, others = drop_high_missing(train_f, [val_f, lab_f, kag_f])
    val_f, lab_f, kag_f = others
    train_f, others, fill_log = fill_missing(train_f, [val_f, lab_f, kag_f])
    val_f, lab_f, kag_f = others
    X_train, X_list, y_train, y_list, fare_bins, dummy_cols, feature_cols = bin_and_dummy(
        train_f, [val_f, lab_f, kag_f]
    )
    X_val, X_test, X_kaggle = X_list
    y_val, y_test, _ = y_list

    age_bins_plot = [0, 12, 18, 35, 50, 80]
    age_labels_plot = ["Child", "Teen", "YoungAdult", "MidAge", "Senior"]
    age_bin_series = pd.cut(train_f["Age"], bins=age_bins_plot, labels=age_labels_plot, include_lowest=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(train_f["Age"], bins=30, kde=True, ax=axes[0], color="#1f77b4")
    axes[0].set_title("训练集 Age（拉格朗日插值后）")
    sns.countplot(x=age_bin_series.astype(str), ax=axes[1], palette="viridis")
    axes[1].set_title("Age 分箱计数")
    axes[1].tick_params(axis="x", rotation=20)
    savefig("14_age_after_impute_bin.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        x=age_bin_series.astype(str), y=train_f["Survived"], ax=ax, palette="coolwarm",
        order=age_labels_plot,
    )
    ax.set_title("训练集：年龄分箱与生存率")
    savefig("26_agebin_survival.png")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy",
    }

    results = {}
    importances = {}
    perm_imp = {}
    grid_rows = []
    best_params = {}
    cv_multi = {}
    kaggle_pred_table = pd.DataFrame({"PassengerId": kaggle_raw["PassengerId"]})

    for name, (est, grid) in build_model_zoo().items():
        print(f"==== GridSearch {name} ====")
        gs = GridSearchCV(
            est, param_grid=grid, cv=cv, scoring=scoring, refit="roc_auc",
            n_jobs=-1, return_train_score=True,
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        pred_tr = best.predict(X_train)
        proba_tr = best.predict_proba(X_train)[:, 1]
        pred_va = best.predict(X_val)
        proba_va = best.predict_proba(X_val)[:, 1]
        pred_te = best.predict(X_test)
        proba_te = best.predict_proba(X_test)[:, 1]
        pred_kg = best.predict(X_kaggle)
        proba_kg = best.predict_proba(X_kaggle)[:, 1]
        kaggle_pred_table[f"{name}_pred"] = pred_kg
        kaggle_pred_table[f"{name}_prob"] = proba_kg

        best_idx = gs.best_index_
        cv_multi[name] = {
            "roc_auc": float(gs.cv_results_["mean_test_roc_auc"][best_idx]),
            "f1": float(gs.cv_results_["mean_test_f1"][best_idx]),
            "precision": float(gs.cv_results_["mean_test_precision"][best_idx]),
            "recall": float(gs.cv_results_["mean_test_recall"][best_idx]),
            "accuracy": float(gs.cv_results_["mean_test_accuracy"][best_idx]),
        }
        results[name] = {
            "train": metrics_dict(y_train, pred_tr, proba_tr),
            "val": metrics_dict(y_val, pred_va, proba_va),
            "test": metrics_dict(y_test, pred_te, proba_te),
            "cv_best_auc": float(gs.best_score_),
            "pred_val": pred_va,
            "proba_val": proba_va,
            "y_val": y_val.values,
            "pred_test": pred_te,
            "proba_test": proba_te,
            "y_test": y_test.values,
            "best_estimator": best,
        }
        best_params[name] = jsonable_params(gs.best_params_)
        print(name, "best", gs.best_params_)
        print("  val ", results[name]["val"])
        print("  test", results[name]["test"])

        cv_df = pd.DataFrame(gs.cv_results_)
        cv_df["model"] = name
        keep = [
            c for c in cv_df.columns
            if c.startswith("param_") or c.startswith("mean_test_") or c in ("rank_test_roc_auc", "model")
        ]
        grid_rows.append(cv_df.sort_values("rank_test_roc_auc").head(8)[keep])

        imp = native_importance(best, X_train.shape[1])
        if imp is not None:
            importances[name] = imp
        pi = permutation_importance(
            best, X_test, y_test, n_repeats=6, random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1
        )
        perm_imp[name] = pd.Series(pi.importances_mean, index=X_train.columns).sort_values(ascending=False)

    plot_split_comparison(results, "val", "val", "验证集")
    plot_split_comparison(results, "test", "test", "测试集（带标签 holdout）")
    plot_extra_model_viz(results, X_train.columns, importances, perm_imp)
    plot_learning_curves(results, X_train, y_train)
    plot_cv_val_test_auc(results)

    names = [n for n in MODEL_ORDER if n in results]
    cv_multi_df = pd.DataFrame(cv_multi).T.loc[names]
    plot_cv_multi_heatmap(cv_multi_df)

    winner = max(names, key=lambda n: (results[n]["val"]["F1"], results[n]["val"]["AUC"]))
    best_test = max(names, key=lambda n: (results[n]["test"]["F1"], results[n]["test"]["AUC"]))

    sub = pd.DataFrame({
        "PassengerId": kaggle_raw["PassengerId"],
        "Survived": kaggle_pred_table[f"{winner}_pred"],
        "SurviveProb": kaggle_pred_table[f"{winner}_prob"],
        "Model": winner,
    })
    sub.to_csv(OUT_DIR / "test_predictions.csv", index=False)
    kaggle_pred_table.to_csv(OUT_DIR / "kaggle_predictions_all_models.csv", index=False)

    labeled_test_out = pd.DataFrame({"PassengerId": labeled_test_part["PassengerId"], "y_true": y_test.values})
    for n in names:
        labeled_test_out[f"{n}_pred"] = results[n]["pred_test"]
        labeled_test_out[f"{n}_prob"] = results[n]["proba_test"]
    labeled_test_out.to_csv(OUT_DIR / "labeled_test_predictions.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(kaggle_pred_table[f"{winner}_prob"], bins=20, ax=axes[0], color="#1f77b4")
    axes[0].set_title(f"Kaggle test.csv 预测概率（{winner}）")
    kaggle_pred_table[f"{winner}_pred"].value_counts().sort_index().plot(
        kind="bar", ax=axes[1], color=["#c0392b", "#27ae60"]
    )
    axes[1].set_title("Kaggle test.csv 0/1 预测计数")
    axes[1].set_xticklabels(["未生存", "生存"], rotation=0)
    savefig("28_kaggle_pred_dist.png")

    val_df = pd.DataFrame({n: results[n]["val"] for n in names}).T
    val_df["CV_AUC"] = [results[n]["cv_best_auc"] for n in names]
    test_df = pd.DataFrame({n: results[n]["test"] for n in names}).T
    train_df = pd.DataFrame({n: results[n]["train"] for n in names}).T
    val_df.to_csv(OUT_DIR / "val_metrics.csv")
    test_df.to_csv(OUT_DIR / "test_metrics.csv")
    train_df.to_csv(OUT_DIR / "train_metrics.csv")
    pd.concat(grid_rows, ignore_index=True).to_csv(OUT_DIR / "gridsearch_top.csv", index=False)
    cv_multi_df.to_csv(OUT_DIR / "cv_multi_metrics.csv")

    def ranks(split_key):
        rows = []
        for metric in ["Precision", "Recall", "F1", "AUC", "Accuracy", "AP"]:
            order = sorted(names, key=lambda n: results[n][split_key][metric], reverse=True)
            rows.append({"metric": metric, **{n: i + 1 for i, n in enumerate(order)}})
        return pd.DataFrame(rows).set_index("metric")

    rank_val = ranks("val")
    rank_test = ranks("test")
    rank_val.to_csv(OUT_DIR / "metric_ranks_val.csv")
    rank_test.to_csv(OUT_DIR / "metric_ranks_test.csv")

    payload = {
        "eda": eda_summary,
        "split": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "labeled_test": int(len(X_test)),
            "kaggle_test": int(len(X_kaggle)),
            "protocol": "train.csv 先 20% 带标签测试，剩余 80% 再 80/20 训练/验证；test.csv 无标签",
            "random_state": RANDOM_STATE,
            "stratify": True,
        },
        "drop_cols": drop_cols,
        "fill_log": fill_log,
        "dummy_cols": dummy_cols,
        "n_features": int(X_train.shape[1]),
        "features": list(X_train.columns),
        "best_params": best_params,
        "val_metrics": {n: results[n]["val"] for n in names},
        "test_metrics": {n: results[n]["test"] for n in names},
        "train_metrics": {n: results[n]["train"] for n in names},
        "cv_best_auc": {n: results[n]["cv_best_auc"] for n in names},
        "cv_multi": cv_multi,
        "winner_by_val_f1": winner,
        "best_on_test_f1": best_test,
        "fare_bins": list(map(float, fare_bins)),
        "title_survival": feat_preview.groupby("Title")["Survived"].mean().to_dict(),
        "sex_survival": train_raw.groupby("Sex")["Survived"].mean().to_dict(),
        "pclass_survival": {str(k): float(v) for k, v in train_raw.groupby("Pclass")["Survived"].mean().items()},
        "kaggle_pred_rate": float(kaggle_pred_table[f"{winner}_pred"].mean()),
    }
    (OUT_DIR / "run_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    write_markdown_report(payload, val_df, test_df, train_df, cv_multi_df, rank_val, rank_test)
    print("DONE val-winner", winner, "test-best", best_test)
    print("VAL\n", val_df)
    print("TEST\n", test_df)
