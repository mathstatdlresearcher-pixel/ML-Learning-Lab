"""三.4 单棵决策树 vs AdaBoost vs Random Forest。"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import CV, FIG_DIR, MODEL_DIR, N_JOBS, RANDOM_STATE, RESULT_DIR, cn
from common.plots import importance_bars, pred_vs_true
from common.style import COLORS, apply_theme, save_fig
from common.utils import print_metrics, reg_metrics, save_csv, save_json
from data_loader.prepare_data import get_boston


def run(quick: bool = False) -> pd.DataFrame:
    apply_theme()
    out = FIG_DIR / "08_boston_ensemble"
    out.mkdir(parents=True, exist_ok=True)
    bundle = get_boston()

    dt_grid = {
        "criterion": ["squared_error"] if quick else ["squared_error", "friedman_mse", "absolute_error"],
        "max_depth": [6, 10] if quick else [5, 8, 11, 14],
        "min_samples_leaf": [2, 6] if quick else [2, 4, 6, 8],
        "min_samples_split": [10, 14] if quick else [10, 14, 18],
    }
    print("  GridSearch DecisionTreeRegressor ...")
    dt_gs = GridSearchCV(DecisionTreeRegressor(random_state=RANDOM_STATE), dt_grid, scoring="r2", cv=CV, n_jobs=N_JOBS)
    dt_gs.fit(bundle.X_train, bundle.y_train)
    dt = dt_gs.best_estimator_

    ada_grid = {
        "n_estimators": [80, 150] if quick else [100, 150, 200, 250],
        "learning_rate": [0.1, 0.2] if quick else [0.05, 0.1, 0.2],
        "loss": ["linear"] if quick else ["linear", "square", "exponential"],
    }
    print("  GridSearch AdaBoostRegressor ...")
    ada_gs = GridSearchCV(
        AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=dt.get_params().get("max_depth", 8), random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
        ),
        ada_grid,
        scoring="r2",
        cv=CV,
        n_jobs=N_JOBS,
    )
    ada_gs.fit(bundle.X_train, bundle.y_train)
    ada = ada_gs.best_estimator_

    rf_path = MODEL_DIR / "boston_random_forest.joblib"
    if rf_path.exists():
        rf = joblib.load(rf_path)
        rf_params = rf.get_params()
    else:
        print("  GridSearch RandomForestRegressor ...")
        rf_gs = GridSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            {
                "n_estimators": [100, 200] if quick else [100, 200, 300],
                "max_depth": [8, None] if quick else [8, 12, None],
                "max_features": ["sqrt", 0.5],
            },
            scoring="r2",
            cv=CV,
            n_jobs=N_JOBS,
        )
        rf_gs.fit(bundle.X_train, bundle.y_train)
        rf = rf_gs.best_estimator_
        rf_params = rf_gs.best_params_
        joblib.dump(rf, rf_path)

    joblib.dump({"dt": dt, "adaboost": ada, "rf": rf}, MODEL_DIR / "boston_ensemble_models.joblib")

    models = {
        "决策树": dt,
        "AdaBoost": ada,
        "随机森林": rf,
    }
    rows = []
    preds_test = {}
    preds_train = {}
    for name, model in models.items():
        tr = reg_metrics(bundle.y_train, model.predict(bundle.X_train))
        te = reg_metrics(bundle.y_test, model.predict(bundle.X_test))
        print_metrics(f"{name} 训练集", tr)
        print_metrics(f"{name} 测试集", te)
        rows.append(
            {
                "model": name,
                **{f"train_{k}": v for k, v in tr.items()},
                **{f"test_{k}": v for k, v in te.items()},
                "overfit_gap_R2": tr["R2"] - te["R2"],
            }
        )
        preds_test[name] = model.predict(bundle.X_test)
        preds_train[name] = model.predict(bundle.X_train)

    table = pd.DataFrame(rows)
    save_csv(table, RESULT_DIR / "boston_ensemble_compare.csv")
    save_json(
        {
            "dt_best_params": dt_gs.best_params_,
            "adaboost_best_params": ada_gs.best_params_,
            "rf_params": {k: rf_params.get(k) for k in ["n_estimators", "max_depth", "max_features", "min_samples_split"] if k in rf_params},
        },
        RESULT_DIR / "boston_ensemble_best_params.json",
    )

    # 指标柱状图
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.5))
    x = np.arange(len(table))
    w = 0.35
    for ax, metric in zip(axes, ["R2", "MSE", "MAE"]):
        ax.bar(x - w / 2, table[f"train_{metric}"], width=w, color=COLORS["slate"], label="训练", edgecolor="white")
        ax.bar(x + w / 2, table[f"test_{metric}"], width=w, color=COLORS["teal"], label="测试", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(table["model"])
        ax.set_title(metric, fontweight="bold")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("单模型 vs 集成：训练/测试指标", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out / "metrics_train_test.png")

    # 过拟合间隙
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.bar(table["model"], table["overfit_gap_R2"], color=[COLORS["coral"], COLORS["teal"], COLORS["cyan"]], edgecolor="white")
    ax.set_ylabel("训练R2 − 测试R2")
    ax.set_title("过拟合间隙（越小越稳）", fontweight="bold")
    for i, v in enumerate(table["overfit_gap_R2"]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out / "overfit_gap.png")

    pred_vs_true(bundle.y_test, preds_test, "测试集：预测 vs 真实房价", out / "pred_vs_true.png")

    # 按真实值排序的预测曲线
    order = np.argsort(bundle.y_test)
    idx = np.arange(len(order))
    fig, axes = plt.subplots(3, 1, figsize=(11.2, 7.6), sharex=True)
    y_sorted = bundle.y_test[order]
    for ax, name in zip(axes, models):
        ax.plot(idx, y_sorted, color=COLORS["slate"], lw=2.2, label="真实值")
        ax.plot(idx, np.asarray(preds_test[name])[order], color=COLORS["teal"], lw=1.7, label=name)
        ax.fill_between(idx, y_sorted, np.asarray(preds_test[name])[order], color=COLORS["teal"], alpha=0.12)
        ax.set_ylabel(name)
        ax.legend(frameon=False, loc="upper left")
    axes[-1].set_xlabel("测试样本（按真实房价排序）")
    fig.suptitle("集成模型对房价曲线的拟合", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out / "sorted_prediction_curves.png")

    # 残差
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    for ax, name in zip(axes, models):
        resid = bundle.y_test - np.asarray(preds_test[name])
        ax.scatter(preds_test[name], resid, s=28, alpha=0.75, c=COLORS["teal"], edgecolors="white", linewidths=0.25)
        ax.axhline(0, color=COLORS["slate"], ls="--")
        ax.set_title(f"{name} 残差", fontweight="bold")
        ax.set_xlabel("预测值")
        ax.set_ylabel("残差")
    fig.tight_layout()
    save_fig(fig, out / "residuals.png")

    importance_bars(bundle.feature_names, dt.feature_importances_, "单棵决策树特征重要性", out / "dt_importance.png")
    importance_bars(bundle.feature_names, ada.feature_importances_, "AdaBoost 特征重要性", out / "adaboost_importance.png")
    importance_bars(bundle.feature_names, rf.feature_importances_, "Random Forest 特征重要性", out / "rf_importance.png")

    # 随 n_estimators 的测试 R2：展示集成把弱学习器抬起来
    n_list = [1, 5, 10, 20, 40, 80, 120] if quick else [1, 5, 10, 20, 40, 80, 120, 200]
    ada_r2, rf_r2 = [], []
    depth = dt.get_params().get("max_depth", 8)
    for n in n_list:
        a = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_STATE),
            n_estimators=n,
            learning_rate=ada_gs.best_params_.get("learning_rate", 0.2),
            random_state=RANDOM_STATE,
        ).fit(bundle.X_train, bundle.y_train)
        r = RandomForestRegressor(n_estimators=n, random_state=RANDOM_STATE, n_jobs=1).fit(bundle.X_train, bundle.y_train)
        ada_r2.append(reg_metrics(bundle.y_test, a.predict(bundle.X_test))["R2"])
        rf_r2.append(reg_metrics(bundle.y_test, r.predict(bundle.X_test))["R2"])
    dt_r2 = reg_metrics(bundle.y_test, dt.predict(bundle.X_test))["R2"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.axhline(dt_r2, color=COLORS["coral"], ls="--", label=f"单棵决策树 R2={dt_r2:.3f}")
    ax.plot(n_list, ada_r2, marker="o", color=COLORS["teal"], label="AdaBoost")
    ax.plot(n_list, rf_r2, marker="o", color=COLORS["amber"], label="Random Forest")
    ax.set_xlabel("基学习器数量 n_estimators")
    ax.set_ylabel("测试集 R2")
    ax.set_title("集成规模增大后相对单模型的增益", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, out / "ensemble_size_vs_single.png")
    save_csv(pd.DataFrame({"n_estimators": n_list, "AdaBoost_test_R2": ada_r2, "RF_test_R2": rf_r2, "DT_test_R2": dt_r2}), RESULT_DIR / "boston_ensemble_size.csv")

    print("  Boston 单模型 vs 集成 完成")
    return table


if __name__ == "__main__":
    run()
