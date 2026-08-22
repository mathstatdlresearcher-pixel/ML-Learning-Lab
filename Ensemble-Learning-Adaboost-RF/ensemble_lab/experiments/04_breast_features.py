"""二.4 AdaBoost(决策树) / Random Forest / Lars 特征重要性对比。"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import Lars, LassoLarsCV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import CV, FIG_DIR, MODEL_DIR, N_JOBS, RANDOM_STATE, RESULT_DIR, cn
from common.plots import importance_bars
from common.style import COLORS, apply_theme, save_fig
from common.utils import clf_metrics, predict_scores, save_csv, save_json, topk_names
from data_loader.prepare_data import get_breast


def _adaboost_dt(bundle, quick: bool):
    path = MODEL_DIR / "adaboost_决策树.joblib"
    if path.exists():
        return joblib.load(path)
    grid = {
        "estimator__max_depth": [1, 3] if quick else [1, 2, 3, 4],
        "n_estimators": [50, 100] if quick else [50, 100, 150, 200],
        "learning_rate": [0.1, 1.0] if quick else [0.05, 0.1, 0.5, 1.0],
    }
    gs = GridSearchCV(
        AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_STATE), random_state=RANDOM_STATE),
        grid,
        scoring="roc_auc",
        cv=CV,
        n_jobs=N_JOBS,
    )
    gs.fit(bundle.X_train, bundle.y_train)
    joblib.dump(gs.best_estimator_, path)
    return gs.best_estimator_


def run(quick: bool = False) -> pd.DataFrame:
    apply_theme()
    out = FIG_DIR / "04_breast_features"
    out.mkdir(parents=True, exist_ok=True)
    bundle = get_breast()
    names = bundle.feature_names

    ada = _adaboost_dt(bundle, quick)
    rf_grid = {
        "n_estimators": [100, 200] if quick else [100, 200, 300],
        "max_depth": [4, None] if quick else [4, 8, None],
        "min_samples_split": [2, 6],
    }
    print("  GridSearch RandomForest ...")
    rf_gs = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        rf_grid,
        scoring="roc_auc",
        cv=CV,
        n_jobs=N_JOBS,
    )
    rf_gs.fit(bundle.X_train, bundle.y_train)
    rf = rf_gs.best_estimator_
    joblib.dump(rf, MODEL_DIR / "breast_random_forest.joblib")

    print("  拟合 Lars / LassoLarsCV ...")
    lars = Lars()
    lars.fit(bundle.X_train_scaled, bundle.y_train)
    lasso = LassoLarsCV(cv=CV).fit(bundle.X_train_scaled, bundle.y_train)
    joblib.dump({"lars": lars, "lasso_lars": lasso}, MODEL_DIR / "breast_lars.joblib")

    ada_imp = np.asarray(ada.feature_importances_, dtype=float)
    rf_imp = np.asarray(rf.feature_importances_, dtype=float)
    lars_imp = np.abs(np.asarray(lars.coef_, dtype=float).ravel())
    lasso_imp = np.abs(np.asarray(lasso.coef_, dtype=float).ravel())

    # 归一化到 0-1 便于并列对比
    def _norm(v):
        s = v.sum()
        return v / s if s > 0 else v

    table = pd.DataFrame(
        {
            "feature": names,
            "feature_cn": [cn(n) for n in names],
            "AdaBoost_DT": ada_imp,
            "RandomForest": rf_imp,
            "Lars_abs_coef": lars_imp,
            "LassoLars_abs_coef": lasso_imp,
            "AdaBoost_DT_norm": _norm(ada_imp),
            "RandomForest_norm": _norm(rf_imp),
            "Lars_norm": _norm(lars_imp),
        }
    )
    table["mean_norm"] = table[["AdaBoost_DT_norm", "RandomForest_norm", "Lars_norm"]].mean(axis=1)
    table = table.sort_values("mean_norm", ascending=False)
    save_csv(table, RESULT_DIR / "breast_feature_importance.csv")

    importance_bars(names, ada_imp, "AdaBoost（决策树基学习器）特征重要性", out / "adaboost_importance.png")
    importance_bars(names, rf_imp, "Random Forest 特征重要性", out / "rf_importance.png")
    importance_bars(names, lars.coef_.ravel(), "Lars 回归系数（可正可负）", out / "lars_coef.png")
    importance_bars(names, lasso.coef_.ravel(), "LassoLarsCV 稀疏系数", out / "lassolars_coef.png")

    topk = 10
    top_ada = set(topk_names(names, ada_imp, topk))
    top_rf = set(topk_names(names, rf_imp, topk))
    top_lars = set(topk_names(names, lars_imp, topk))
    common3 = sorted(top_ada & top_rf & top_lars)
    common_tree = sorted(top_ada & top_rf)

    # 三模型归一化重要性并列
    top_show = table.head(12)
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    y = np.arange(len(top_show))
    h = 0.25
    ax.barh(y - h, top_show["AdaBoost_DT_norm"], height=h, color=COLORS["teal"], label="AdaBoost-DT", edgecolor="white")
    ax.barh(y, top_show["RandomForest_norm"], height=h, color=COLORS["coral"], label="Random Forest", edgecolor="white")
    ax.barh(y + h, top_show["Lars_norm"], height=h, color=COLORS["amber"], label="Lars", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(top_show["feature_cn"])
    ax.invert_yaxis()
    ax.set_xlabel("归一化重要性")
    ax.set_title("三类模型特征重要性对比（Top12 综合）", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_fig(fig, out / "importance_compare.png")

    # 模型本身分类表现（便于报告）
    metrics_rows = []
    for name, model, Xtr, Xte in [
        ("AdaBoost-DT", ada, bundle.X_train, bundle.X_test),
        ("RandomForest", rf, bundle.X_train, bundle.X_test),
    ]:
        pred = model.predict(Xte)
        score = predict_scores(model, Xte)
        m = clf_metrics(bundle.y_test, pred, score)
        metrics_rows.append({"model": name, **m})
    # Lars 作回归阈值 0.5
    lars_score = lars.predict(bundle.X_test_scaled)
    lars_pred = (lars_score >= 0.5).astype(int)
    metrics_rows.append({"model": "Lars(阈值0.5)", **clf_metrics(bundle.y_test, lars_pred, lars_score)})
    save_csv(pd.DataFrame(metrics_rows), RESULT_DIR / "breast_feature_models_metrics.csv")

    conclusion = {
        "rf_best_params": rf_gs.best_params_,
        "top10_AdaBoost": [cn(x) for x in topk_names(names, ada_imp, 10)],
        "top10_RandomForest": [cn(x) for x in topk_names(names, rf_imp, 10)],
        "top10_Lars": [cn(x) for x in topk_names(names, lars_imp, 10)],
        "intersection_all_three": [cn(x) for x in common3],
        "intersection_two_trees": [cn(x) for x in common_tree],
        "key_factors": [cn(x) for x in table.head(8)["feature"].tolist()],
        "note": (
            "树模型重要性来自分裂增益加权；Lars 重要性取回归系数绝对值。"
            "三者交叉或综合排名靠前的特征，可视为影响乳腺癌分类的关键因素。"
        ),
    }
    save_json(conclusion, RESULT_DIR / "breast_key_factors.json")
    print("  三模型共同 Top 特征:", conclusion["intersection_all_three"] or "(无完全交集，见综合排名)")
    print("  综合关键因素:", conclusion["key_factors"])
    return table


if __name__ == "__main__":
    run()
