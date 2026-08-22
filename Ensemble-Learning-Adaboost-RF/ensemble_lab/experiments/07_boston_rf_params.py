"""三.3 Random Forest 参数对 R2 / MSE / MAE 的影响。"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import CV, FIG_DIR, MODEL_DIR, N_JOBS, RANDOM_STATE, RESULT_DIR
from common.plots import line_metrics
from common.style import COLORS, apply_theme, save_fig
from common.utils import print_metrics, reg_metrics, save_csv, save_json
from data_loader.prepare_data import get_boston


def _eval_rf(params, Xtr, ytr, Xte, yte) -> dict:
    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1, **params)
    model.fit(Xtr, ytr)
    tr = reg_metrics(ytr, model.predict(Xtr))
    te = reg_metrics(yte, model.predict(Xte))
    return {f"train_{k}": v for k, v in tr.items()} | {f"test_{k}": v for k, v in te.items()}


def _sweep(name, values, base, bundle, stringify=False) -> pd.DataFrame:
    rows = []
    for v in values:
        params = dict(base)
        params[name] = v
        m = _eval_rf(params, bundle.X_train, bundle.y_train, bundle.X_test, bundle.y_test)
        rows.append({"param": name, "value": str(v) if stringify else v, **m})
    return pd.DataFrame(rows)


def _three_metric_plot(df: pd.DataFrame, xlabel: str, title_prefix: str, save_path: Path, x_as_str=False):
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))
    xs = df["value"].tolist()
    for ax, metric in zip(axes, ["R2", "MSE", "MAE"]):
        ax.plot(xs, df[f"train_{metric}"], marker="o", color=COLORS["slate"], label="训练集")
        ax.plot(xs, df[f"test_{metric}"], marker="o", color=COLORS["teal"], label="测试集")
        ax.set_title(metric, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.legend(frameon=False, fontsize=9)
        if x_as_str:
            ax.set_xticks(xs)
            ax.set_xticklabels([str(v) for v in xs], rotation=20)
    fig.suptitle(title_prefix, fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, save_path)


def run(quick: bool = False) -> pd.DataFrame:
    apply_theme()
    out = FIG_DIR / "07_boston_rf_params"
    out.mkdir(parents=True, exist_ok=True)
    bundle = get_boston()

    base = {"n_estimators": 150, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"}

    n_est = [20, 50, 100] if quick else [10, 30, 50, 100, 200, 300]
    depths = [2, 6, None] if quick else [2, 4, 6, 8, 10, 14, None]
    max_feat = ["sqrt", 1.0] if quick else ["sqrt", "log2", 0.3, 0.5, 0.8, 1.0]
    splits = [2, 10] if quick else [2, 5, 10, 20, 40]
    leaves = [1, 4] if quick else [1, 2, 4, 8, 12]

    sweeps = {
        "n_estimators": _sweep("n_estimators", n_est, base, bundle),
        "max_depth": _sweep("max_depth", depths, base, bundle, stringify=True),
        "max_features": _sweep("max_features", max_feat, base, bundle, stringify=True),
        "min_samples_split": _sweep("min_samples_split", splits, base, bundle),
        "min_samples_leaf": _sweep("min_samples_leaf", leaves, base, bundle),
    }
    all_df = pd.concat(sweeps.values(), ignore_index=True)
    save_csv(all_df, RESULT_DIR / "boston_rf_param_sweeps.csv")

    _three_metric_plot(sweeps["n_estimators"], "n_estimators", "n_estimators 对 RF 性能的影响", out / "sweep_n_estimators.png")
    _three_metric_plot(sweeps["max_depth"], "max_depth", "max_depth 对 RF 性能的影响", out / "sweep_max_depth.png", x_as_str=True)
    _three_metric_plot(sweeps["max_features"], "max_features", "max_features 对 RF 性能的影响", out / "sweep_max_features.png", x_as_str=True)
    _three_metric_plot(sweeps["min_samples_split"], "min_samples_split", "min_samples_split 对 RF 性能的影响", out / "sweep_min_samples_split.png")
    _three_metric_plot(sweeps["min_samples_leaf"], "min_samples_leaf", "min_samples_leaf 对 RF 性能的影响", out / "sweep_min_samples_leaf.png")

    grid = {
        "n_estimators": [100, 200] if quick else [100, 200, 300],
        "max_depth": [6, None] if quick else [6, 10, 14, None],
        "max_features": ["sqrt", 0.5] if quick else ["sqrt", 0.5, 1.0],
        "min_samples_split": [2, 8] if quick else [2, 5, 10],
    }
    print("  GridSearch RandomForestRegressor ...")
    gs = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        grid,
        scoring="r2",
        cv=CV,
        n_jobs=N_JOBS,
        refit=True,
    )
    gs.fit(bundle.X_train, bundle.y_train)
    best = gs.best_estimator_
    joblib.dump(best, MODEL_DIR / "boston_random_forest.joblib")
    cv_df = pd.DataFrame(gs.cv_results_)
    keep = [c for c in cv_df.columns if c.startswith("param_") or c in {"mean_test_score", "std_test_score", "rank_test_score"}]
    save_csv(cv_df[keep].sort_values("rank_test_score"), RESULT_DIR / "boston_rf_gridcv.csv")

    test_m = reg_metrics(bundle.y_test, best.predict(bundle.X_test))
    train_m = reg_metrics(bundle.y_train, best.predict(bundle.X_train))
    print_metrics("RF 最优 测试集", test_m)
    save_json({"best_params": gs.best_params_, "cv_R2": float(gs.best_score_), "test": test_m, "train": train_m}, RESULT_DIR / "boston_rf_best.json")

    # 最优模型下 n_estimators 学习曲线
    n_curve = [10, 20, 40, 80, 120, 200] if quick else [10, 20, 40, 80, 120, 160, 220, 300]
    bp = dict(gs.best_params_)
    bp.pop("n_estimators", None)
    grow = _sweep("n_estimators", n_curve, bp, bundle)
    save_csv(grow, RESULT_DIR / "boston_rf_n_estimators_learning.csv")
    line_metrics(
        grow["value"],
        {"训练R2": grow["train_R2"], "测试R2": grow["test_R2"]},
        "n_estimators",
        "最优其他参数下，树的数量对 R2 的影响",
        out / "learning_n_estimators_r2.png",
    )
    print("  Boston RF 参数实验完成")
    return all_df


if __name__ == "__main__":
    run()
