"""实验 03：LARS 变量选择稳定性。"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lars, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FIG_DIR, N_REPEATS, REPORT_DIR, TEST_SIZE
from data.generate_data import build_simulated_datasets
from models.regressors import fit_ols
from utils.metrics import evaluate, summarize_repeats
from utils.style import apply_theme
from utils.viz import plot_lars_stability, plot_metric_bars


def _lars_select(X_train, y_train, candidate_ks=None):
    """CV 选择最优非零个数，再拟合 LARS。"""
    p = X_train.shape[1]
    if candidate_ks is None:
        candidate_ks = sorted(set([2, 3, 4, 5, min(8, p), p]))
    candidate_ks = [k for k in candidate_ks if 1 <= k <= p]

    best_k, best_score = candidate_ks[0], np.inf
    for k in candidate_ks:
        est = Lars(n_nonzero_coefs=k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = -cross_val_score(
                est, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
            ).mean()
        if score < best_score:
            best_score, best_k = float(score), k

    model = Lars(n_nonzero_coefs=best_k)
    model.fit(X_train, y_train)
    mask = np.zeros(p, dtype=bool)
    active = list(getattr(model, "active_", []))
    if active:
        for idx in active[:best_k]:
            mask[int(idx)] = True
    else:
        mask = np.abs(np.asarray(model.coef_).ravel()) > 1e-10
    if mask.sum() < best_k:
        order = np.argsort(-np.abs(np.asarray(model.coef_).ravel()))
        for idx in order[:best_k]:
            mask[int(idx)] = True
    return mask, best_k


def _run_stability(bundle, n_repeats: int = N_REPEATS) -> dict:
    # 稳定性图只展示相关特征 + 前 4 个 noise，避免图过宽
    show_names = list(bundle.relevant_features)
    noise_cols = [c for c in bundle.feature_names if str(c).startswith("noise")][:4]
    show_names = show_names + noise_cols
    show_idx = [bundle.feature_names.index(n) for n in show_names]

    p = len(bundle.feature_names)
    selection_full = np.zeros((n_repeats, p), dtype=int)
    before_rows, after_rows = [], []
    ks = []

    for i in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            bundle.X.values, bundle.y.values, test_size=TEST_SIZE, random_state=2000 + i
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        full = fit_ols(X_train_s, y_train, X_test_s, y_test)
        before_rows.append(full.metrics)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask, k = _lars_select(X_train_s, y_train)
        ks.append(k)
        selection_full[i] = mask.astype(int)

        if mask.sum() == 0:
            after_rows.append(full.metrics)
            continue
        lr = LinearRegression().fit(X_train_s[:, mask], y_train)
        after_rows.append(evaluate(y_test, lr.predict(X_test_s[:, mask])))

    selection = selection_full[:, show_idx]
    return {
        "selection_matrix": selection,
        "feature_names": show_names,
        "selection_freq": {
            n: float(selection_full[:, bundle.feature_names.index(n)].mean())
            for n in show_names
        },
        "mean_k": float(np.mean(ks)),
        "before": summarize_repeats(before_rows),
        "after": summarize_repeats(after_rows),
    }


def run() -> dict:
    apply_theme()
    out_dir = FIG_DIR / "03_lars_stability"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles = build_simulated_datasets()
    targets = ["y2_noisy", "y3_no_multicollinearity", "y4_multicollinearity"]
    report = {}

    for key in targets:
        bundle = bundles[key]
        print("-" * 60)
        print(f"LARS stability: {key}")
        res = _run_stability(bundle)
        plot_lars_stability(
            res["selection_matrix"],
            res["feature_names"],
            title=f"{key} LARS 选择稳定性",
            save_path=out_dir / f"{key}_stability.png",
        )
        compare = {
            "全变量OLS": {
                "R2": res["before"]["R2"]["mean"],
                "RMSE": res["before"]["RMSE"]["mean"],
                "MAE": res["before"]["MAE"]["mean"],
            },
            "LARS后OLS": {
                "R2": res["after"]["R2"]["mean"],
                "RMSE": res["after"]["RMSE"]["mean"],
                "MAE": res["after"]["MAE"]["mean"],
            },
        }
        plot_metric_bars(
            compare,
            metrics=("R2", "RMSE", "MAE"),
            title=f"{key} 选择前后",
            save_path=out_dir / f"{key}_before_after.png",
        )

        patterns = [tuple(row.tolist()) for row in res["selection_matrix"]]
        report[key] = {
            "selection_freq": res["selection_freq"],
            "mean_k": res["mean_k"],
            "unique_patterns": len(set(patterns)),
            "metrics_before": res["before"],
            "metrics_after": res["after"],
        }
        print(f"  mean_k={res['mean_k']:.2f}")
        print(f"  freq={res['selection_freq']}")
        print(
            f"  R2 {res['before']['R2']['mean']:.4f} -> {res['after']['R2']['mean']:.4f} | "
            f"RMSE {res['before']['RMSE']['mean']:.4f} -> {res['after']['RMSE']['mean']:.4f}"
        )

    with open(REPORT_DIR / "03_lars_stability.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("=" * 60)
    print("[03] done")
    return report


if __name__ == "__main__":
    run()
