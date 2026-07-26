"""实验 02：模拟数据模型对比。"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FIG_DIR, N_REPEATS, REPORT_DIR, TEST_SIZE
from data.generate_data import build_simulated_datasets, train_test_scale
from models.regressors import fit_all_models
from utils.metrics import format_metrics, summarize_repeats
from utils.style import apply_theme
from utils.viz import (
    plot_alpha_path,
    plot_coef_compare,
    plot_coef_stability,
    plot_metric_bars,
    plot_pred_vs_true,
    plot_residuals,
    plot_series_compare,
)


def _true_coef_on_scaled(bundle) -> dict:
    out = {"intercept": float(bundle.true_coef.get("intercept", 0.0))}
    for name in bundle.feature_names:
        out[name] = float(bundle.true_coef.get(name, 0.0) * bundle.X[name].std(ddof=0))
    return out


def _fit_once(bundle, random_state: int):
    X_train, X_test, y_train, y_test, _ = train_test_scale(
        bundle.X, bundle.y, test_size=TEST_SIZE, random_state=random_state, scale=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = fit_all_models(
            X_train, y_train, X_test, y_test, feature_names=bundle.feature_names
        )
    return results, y_test


def run() -> dict:
    apply_theme()
    out_dir = FIG_DIR / "02_sim_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles = build_simulated_datasets()
    all_rows = []
    report = {}

    for key, bundle in bundles.items():
        print("-" * 60)
        print(f"{key} | {bundle.description}")

        results, y_test = _fit_once(bundle, random_state=42)
        metrics_map = {n: r.metrics for n, r in results.items()}
        preds = {n: r.y_pred for n, r in results.items()}
        true_scaled = _true_coef_on_scaled(bundle)

        plot_metric_bars(
            metrics_map,
            metrics=("R2", "RMSE", "MAE"),
            title=f"{key} 指标对比",
            save_path=out_dir / f"{key}_metrics.png",
        )
        plot_pred_vs_true(y_test, preds, title=f"{key} 预测对比", save_path=out_dir / f"{key}_pred_vs_true.png")
        plot_series_compare(y_test, preds, title=f"{key} 测试曲线", save_path=out_dir / f"{key}_series.png")
        plot_residuals(y_test, results["OLS"].y_pred, title=f"{key} OLS 残差", save_path=out_dir / f"{key}_residuals.png")

        rel = bundle.relevant_features
        est = {n: r.coef for n, r in results.items()}
        plot_coef_compare(
            rel,
            true_scaled,
            est,
            all_feature_names=bundle.feature_names,
            title=f"{key} 相关特征系数",
            save_path=out_dir / f"{key}_coef.png",
        )

        if results["Ridge"].extra.get("cv_mse") is not None:
            plot_alpha_path(
                results["Ridge"].extra["cv_alphas"],
                results["Ridge"].extra["cv_mse"],
                results["Ridge"].best_alpha,
                title=f"{key} RidgeCV",
                save_path=out_dir / f"{key}_ridge_path.png",
            )

        repeat_rows = {n: [] for n in results}
        coef_runs = {n: [] for n in results}
        for i in range(N_REPEATS):
            rs, _ = _fit_once(bundle, random_state=1000 + i)
            for name, fr in rs.items():
                row = {"dataset": key, "model": name, "repeat": i, **fr.metrics}
                if fr.best_alpha is not None:
                    row["alpha"] = fr.best_alpha
                if fr.selected_features is not None:
                    row["n_selected"] = len(fr.selected_features)
                repeat_rows[name].append(row)
                all_rows.append(row)
                coef_runs[name].append(fr.coef)

        coef_arr = {n: np.vstack(v) for n, v in coef_runs.items()}
        rel_idx = [bundle.feature_names.index(f) for f in rel]
        plot_coef_stability(
            rel,
            coef_arr,
            feature_index=rel_idx,
            title=f"{key} 系数稳定性",
            save_path=out_dir / f"{key}_coef_stability.png",
        )

        summary = {n: summarize_repeats(rows) for n, rows in repeat_rows.items()}
        report[key] = {
            "description": bundle.description,
            "noise_std": bundle.noise_std,
            "relevant_features": rel,
            "single_split": {
                n: {
                    "metrics": fr.metrics,
                    "selected": fr.selected_features,
                    "alpha": fr.best_alpha,
                }
                for n, fr in results.items()
            },
            "repeats_mean_std": summary,
        }

        print("  single split:")
        for n, fr in results.items():
            sel = f"  sel={fr.selected_features}" if fr.selected_features is not None else ""
            print(f"    {n:6s}: {format_metrics(fr.metrics)}{sel}")
        print(f"  {N_REPEATS}-run mean R2 / RMSE:")
        for n, s in summary.items():
            print(f"    {n:6s}: R2={s['R2']['mean']:.4f}  RMSE={s['RMSE']['mean']:.4f}")

    pd.DataFrame(all_rows).to_csv(REPORT_DIR / "02_sim_repeats.csv", index=False, encoding="utf-8-sig")
    with open(REPORT_DIR / "02_sim_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("[02] done")
    return report


if __name__ == "__main__":
    run()
