"""二.3 不同基学习器下的 AdaBoost：网格搜索 + Precision/Recall/F1/AUC。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import CV, FIG_DIR, MODEL_DIR, N_JOBS, RANDOM_STATE, RESULT_DIR
from common.plots import confusion_grid, metric_bars, roc_overlay
from common.style import apply_theme
from common.utils import clf_metrics, print_metrics, predict_scores, save_csv, save_json
from data_loader.prepare_data import get_breast


def _grids(quick: bool) -> dict:
    if quick:
        return {
            "决策树": {
                "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
                "X": "raw",
                "param_grid": {
                    "estimator__max_depth": [1, 3],
                    "n_estimators": [50, 100],
                    "learning_rate": [0.1, 1.0],
                },
            },
            "逻辑回归": {
                "estimator": LogisticRegression(max_iter=400, solver="lbfgs"),
                "X": "scaled",
                "param_grid": {
                    "estimator__C": [0.1, 1.0],
                    "n_estimators": [40, 80],
                    "learning_rate": [0.1, 0.5],
                },
            },
            "SVM": {
                "estimator": SVC(kernel="linear"),
                "X": "scaled",
                "param_grid": {
                    "estimator__C": [0.5, 2.0],
                    "n_estimators": [10, 20],
                    "learning_rate": [0.5, 1.0],
                },
            },
        }
    return {
        "决策树": {
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "X": "raw",
            "param_grid": {
                "estimator__max_depth": [1, 2, 3, 4],
                "n_estimators": [50, 100, 150, 200],
                "learning_rate": [0.05, 0.1, 0.5, 1.0],
            },
        },
        "逻辑回归": {
            "estimator": LogisticRegression(max_iter=500, solver="lbfgs"),
            "X": "scaled",
            "param_grid": {
                "estimator__C": [0.1, 1.0, 10.0],
                "n_estimators": [40, 80, 120],
                "learning_rate": [0.1, 0.5, 1.0],
            },
        },
        "SVM": {
            "estimator": SVC(kernel="linear"),
            "X": "scaled",
            "param_grid": {
                "estimator__C": [0.1, 1.0, 10.0],
                "n_estimators": [10, 20, 30],
                "learning_rate": [0.1, 0.5, 1.0],
            },
        },
    }


def _xy(bundle, kind: str):
    if kind == "scaled":
        return bundle.X_train_scaled, bundle.X_test_scaled
    return bundle.X_train, bundle.X_test


def run(quick: bool = False) -> pd.DataFrame:
    apply_theme()
    out = FIG_DIR / "03_breast_adaboost"
    out.mkdir(parents=True, exist_ok=True)
    bundle = get_breast()

    rows = []
    scores = {}
    preds = {}
    best_models = {}

    for name, spec in _grids(quick).items():
        Xtr, Xte = _xy(bundle, spec["X"])
        model = AdaBoostClassifier(
            estimator=spec["estimator"],
            random_state=RANDOM_STATE,
        )
        gs = GridSearchCV(
            model,
            spec["param_grid"],
            scoring="roc_auc",
            cv=CV,
            n_jobs=N_JOBS,
            refit=True,
        )
        t0 = time.time()
        print(f"  GridSearch AdaBoost[{name}] ...")
        gs.fit(Xtr, bundle.y_train)
        elapsed = time.time() - t0
        best = gs.best_estimator_
        y_pred = best.predict(Xte)
        y_score = predict_scores(best, Xte)
        m = clf_metrics(bundle.y_test, y_pred, y_score)
        m_train = clf_metrics(bundle.y_train, best.predict(Xtr), predict_scores(best, Xtr))
        print_metrics(f"AdaBoost-{name} 测试集", m)
        print(f"    best_params={gs.best_params_}  cv_auc={gs.best_score_:.4f}  time={elapsed:.1f}s")

        cv_df = pd.DataFrame(gs.cv_results_)
        keep = [c for c in cv_df.columns if c.startswith("param_") or c in {"mean_test_score", "std_test_score", "rank_test_score"}]
        save_csv(cv_df[keep].sort_values("rank_test_score"), RESULT_DIR / f"breast_adaboost_{name}_cv.csv")
        joblib.dump(best, MODEL_DIR / f"adaboost_{name}.joblib")

        row = {
            "base_learner": name,
            "best_params": str(gs.best_params_),
            "cv_AUC": float(gs.best_score_),
            "search_seconds": float(elapsed),
            **{f"test_{k}": v for k, v in m.items()},
            **{f"train_{k}": v for k, v in m_train.items()},
        }
        rows.append(row)
        scores[f"AdaBoost-{name}"] = y_score
        preds[f"AdaBoost-{name}"] = y_pred
        best_models[name] = best

        save_json({"best_params": gs.best_params_, "cv_AUC": gs.best_score_, "test": m, "train": m_train}, RESULT_DIR / f"breast_adaboost_{name}_best.json")

    table = pd.DataFrame(rows)
    save_csv(table, RESULT_DIR / "breast_adaboost_compare.csv")

    metric_map = {
        r["base_learner"]: {
            "Precision": r["test_Precision"],
            "Recall": r["test_Recall"],
            "F1": r["test_F1"],
            "AUC": r["test_AUC"],
        }
        for r in rows
    }
    metric_bars(metric_map, ["Precision", "Recall", "F1", "AUC"], "不同基学习器 AdaBoost 测试集对比", out / "metrics_compare.png")
    roc_overlay(bundle.y_test, scores, "AdaBoost 不同基学习器 ROC", out / "roc_compare.png")
    confusion_grid(bundle.y_test, preds, ["良性", "恶性"], "AdaBoost 混淆矩阵（测试集）", out / "confusion_compare.png")

    # 决策树网格热力图：n_estimators × learning_rate（取最佳 max_depth 切片）
    cv_dt = pd.read_csv(RESULT_DIR / "breast_adaboost_决策树_cv.csv")
    if "param_estimator__max_depth" in cv_dt.columns:
        best_depth = table.loc[table["base_learner"] == "决策树", "best_params"].iloc[0]
        # 直接用 rank=1 的 depth
        depth_col = "param_estimator__max_depth"
        top_depth = cv_dt.sort_values("rank_test_score").iloc[0][depth_col]
        sub = cv_dt[cv_dt[depth_col] == top_depth]
        pivot = sub.pivot_table(index="param_n_estimators", columns="param_learning_rate", values="mean_test_score")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from common.style import save_fig

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn", ax=ax)
        ax.set_title(f"AdaBoost-决策树 CV-AUC（max_depth={top_depth}）", fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out / "dt_grid_heatmap.png")

    print("  Breast AdaBoost 对比完成")
    return table


if __name__ == "__main__":
    run()
