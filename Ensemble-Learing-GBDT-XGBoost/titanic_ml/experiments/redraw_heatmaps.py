"""Redraw heatmaps from saved CSVs (no grid search). python -m titanic_ml.experiments.redraw_heatmaps"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from titanic_ml.config.settings import MODEL_ORDER, OUT_DIR
from titanic_ml.data.loading import load_raw
from titanic_ml.visualization.eda_plots import plot_eda
from titanic_ml.visualization.heatmaps import annotated_heatmap
from titanic_ml.visualization.model_plots import plot_cv_multi_heatmap
from titanic_ml.visualization.style import apply_plot_style, savefig


def _metric_heatmap(csv_name: str, png: str, title: str, extra_drop=()) -> None:
    df = pd.read_csv(OUT_DIR / csv_name, index_col=0)
    df = df.drop(columns=[c for c in extra_drop if c in df.columns], errors="ignore")
    order = [m for m in MODEL_ORDER if m in df.index]
    cols = [c for c in ["Precision", "Recall", "F1", "AUC", "Accuracy"] if c in df.columns]
    heat = df.loc[order, cols].T
    heat.index.name = "指标"
    fig, ax = plt.subplots(figsize=(14, 6.6))
    annotated_heatmap(heat, ax, fmt=".3f", cmap="YlGnBu", vmin=0.65, vmax=1.0, annot_size=12, x_rotation=0)
    ax.set_xlabel("模型")
    ax.set_ylabel("指标")
    ax.set_title(title)
    savefig(png)


def main() -> None:
    apply_plot_style()
    train_raw, _ = load_raw()
    plot_eda(train_raw)
    _metric_heatmap("val_metrics.csv", "val_metrics_heatmap.png", "验证集指标热力图", extra_drop=("CV_AUC", "AP"))
    _metric_heatmap("test_metrics.csv", "test_metrics_heatmap.png", "测试集（带标签 holdout）指标热力图", extra_drop=("AP",))
    cv = pd.read_csv(OUT_DIR / "cv_multi_metrics.csv", index_col=0)
    plot_cv_multi_heatmap(cv)
    print("heatmaps redrawn")


if __name__ == "__main__":
    main()
