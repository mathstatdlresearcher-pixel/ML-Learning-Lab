"""
实验脚本 05：Boston 房价上 OLS / Ridge / Lasso / LARS 建模与对比
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FIG_DIR, N_REPEATS, REPORT_DIR, TEST_SIZE
from data.boston import boston_train_test, preprocess_boston
from data.generate_data import train_test_scale
from models.regressors import fit_all_models
from utils.metrics import format_metrics, summarize_repeats
from utils.style import apply_theme
from utils.viz import (
    plot_alpha_path,
    plot_feature_importance,
    plot_metric_bars,
    plot_pred_vs_true,
    plot_residuals,
    plot_series_compare,
    plot_series_one_vs_true,
)


def run() -> dict:
    apply_theme()
    out_dir = FIG_DIR / "05_boston_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, X_train, X_test, y_train, y_test, scaler, info = boston_train_test(scale=True)
    feature_names = list(X.columns)

    results = fit_all_models(X_train, y_train, X_test, y_test, feature_names=feature_names)
    metrics_map = {name: r.metrics for name, r in results.items()}
    preds = {name: r.y_pred for name, r in results.items()}

    plot_metric_bars(
        metrics_map,
        metrics=("R2", "RMSE", "MAE"),
        title="Boston 模型指标对比",
        save_path=out_dir / "metrics.png",
    )
    plot_pred_vs_true(y_test, preds, title="Boston 真实值 vs 预测值", save_path=out_dir / "pred_vs_true.png")
    plot_series_compare(y_test, preds, title="Boston 测试集预测曲线（总览）", save_path=out_dir / "series.png")
    # 每个模型单独一张：预测 vs 真实
    for name, y_pred in preds.items():
        plot_series_one_vs_true(
            y_test,
            y_pred,
            model_name=name,
            title=f"Boston 测试集：{name} vs 真实值",
            save_path=out_dir / f"series_{name.lower()}.png",
        )
    plot_residuals(y_test, results["OLS"].y_pred, title="Boston OLS 残差", save_path=out_dir / "residuals_ols.png")

    for name in ["OLS", "Ridge", "Lasso", "LARS"]:
        plot_feature_importance(
            feature_names,
            results[name].coef,
            title=f"Boston · {name} 系数",
            save_path=out_dir / f"coef_{name.lower()}.png",
        )

    if results["Ridge"].extra.get("cv_mse") is not None:
        plot_alpha_path(
            results["Ridge"].extra["cv_alphas"],
            results["Ridge"].extra["cv_mse"],
            results["Ridge"].best_alpha,
            title="Boston · RidgeCV 路径",
            save_path=out_dir / "ridge_path.png",
        )

    # 多次重复
    repeat_rows = []
    X_df, y_s, _ = preprocess_boston()
    for i in range(N_REPEATS):
        Xtr, Xte, ytr, yte, _ = train_test_scale(X_df, y_s, test_size=TEST_SIZE, random_state=3000 + i)
        rs = fit_all_models(Xtr, ytr, Xte, yte, feature_names=list(X_df.columns))
        for name, fr in rs.items():
            repeat_rows.append({"model": name, "repeat": i, **fr.metrics})

    rep_df = pd.DataFrame(repeat_rows)
    rep_df.to_csv(REPORT_DIR / "05_boston_repeats.csv", index=False, encoding="utf-8-sig")
    summary_rep = {
        m: summarize_repeats(rep_df[rep_df["model"] == m].to_dict("records"))
        for m in rep_df["model"].unique()
    }

    report = {
        "selected_features": feature_names,
        "preprocess": info,
        "single_split": {
            name: {
                "metrics": fr.metrics,
                "coef": fr.coef.tolist(),
                "intercept": fr.intercept,
                "alpha": fr.best_alpha,
                "selected": fr.selected_features,
            }
            for name, fr in results.items()
        },
        "repeats_mean_std": summary_rep,
    }
    with open(REPORT_DIR / "05_boston_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("【05】Boston 建模完成")
    print(f"特征: {feature_names}")
    for name, fr in results.items():
        extra = f"  α={fr.best_alpha:.4g}" if fr.best_alpha is not None else ""
        sel = f"  选中={fr.selected_features}" if fr.selected_features is not None else ""
        print(f"  {name:6s}: {format_metrics(fr.metrics)}{extra}{sel}")
    print(f"{N_REPEATS} 次重复均值 R²:")
    for name, s in summary_rep.items():
        print(f"  {name:6s}: {s['R2']['mean']:.4f} ± {s['R2']['std']:.4f}")
    print(f"图片目录: {out_dir}")
    return report


if __name__ == "__main__":
    run()
