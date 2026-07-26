"""实验 01：模拟数据 EDA。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FIG_DIR, REPORT_DIR
from data.generate_data import (
    build_simulated_datasets,
    compute_vif,
    generate_raw_features,
    save_simulated_csvs,
)
from utils.style import apply_theme
from utils.viz import (
    plot_3d_scatter,
    plot_corr_heatmap,
    plot_pair_overview,
    plot_vif_bars,
    plot_xy_scatter,
)


def run() -> dict:
    apply_theme()
    out_dir = FIG_DIR / "01_eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = save_simulated_csvs()
    bundles = build_simulated_datasets()
    feat_df = pd.DataFrame(generate_raw_features())

    plot_corr_heatmap(feat_df, title="特征相关矩阵 x1-x5", save_path=out_dir / "corr_features.png")

    vif4 = compute_vif(bundles["y4_multicollinearity"].X)
    vif3 = compute_vif(bundles["y3_no_multicollinearity"].X)
    vif4.to_csv(REPORT_DIR / "vif_y4.csv", index=False, encoding="utf-8-sig")
    plot_vif_bars(vif4, title="y4 多重共线性 VIF", save_path=out_dir / "vif_y4.png")
    plot_vif_bars(vif3, title="y3 无多重共线性 VIF", save_path=out_dir / "vif_y3.png")

    b1 = bundles["y1_no_noise"]
    b2 = bundles["y2_noisy"]
    plot_3d_scatter(b1.X["x1"], b1.X["x2"], b1.y, "y1 三维散点", out_dir / "y1_3d.png")
    plot_3d_scatter(b2.X["x1"], b2.X["x2"], b2.y, "y2 三维散点", out_dir / "y2_3d.png")
    plot_xy_scatter(b1.X["x1"], b1.y, "y1 线性关系 x1-y", "x1", save_path=out_dir / "y1_x1.png")
    plot_xy_scatter(b2.X["x1"], b2.y, "y2 线性关系 x1-y", "x1", save_path=out_dir / "y2_x1.png")
    plot_xy_scatter(b2.X["x2"], b2.y, "y2 线性关系 x2-y", "x2", save_path=out_dir / "y2_x2.png")

    df4 = bundles["y4_multicollinearity"].X[bundles["y4_multicollinearity"].relevant_features].copy()
    df4["y4"] = bundles["y4_multicollinearity"].y.values
    plot_pair_overview(
        df4,
        features=bundles["y4_multicollinearity"].relevant_features,
        target="y4",
        title="y4 相关特征配对",
        save_path=out_dir / "y4_pairplot.png",
    )

    meta = {
        k: {
            "description": b.description,
            "noise_std": b.noise_std,
            "signal_std": b.signal_std,
            "shape": list(b.X.shape),
            "relevant": b.relevant_features,
        }
        for k, b in bundles.items()
    }
    with open(REPORT_DIR / "01_eda_summary.json", "w", encoding="utf-8") as f:
        json.dump({"datasets": meta, "csv": {k: str(v) for k, v in paths.items()}}, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("[01] EDA done")
    for k, m in meta.items():
        print(f"  {k}: shape={m['shape']}, σε={m['noise_std']:.3f}")
    return meta


if __name__ == "__main__":
    run()
