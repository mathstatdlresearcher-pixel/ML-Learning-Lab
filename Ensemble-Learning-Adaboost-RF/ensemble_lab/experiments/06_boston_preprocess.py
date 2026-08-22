"""三.2 Boston 房价数据预处理。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import FIG_DIR, RESULT_DIR, cn
from common.style import COLORS, apply_theme, save_fig
from common.utils import save_csv, save_json
from data_loader.prepare_data import load_boston_raw, prepare_boston


def run() -> dict:
    apply_theme()
    out = FIG_DIR / "06_boston_preprocess"
    out.mkdir(parents=True, exist_ok=True)

    raw = load_boston_raw()
    bundle = prepare_boston(drop_capped=False)

    steps = pd.DataFrame(
        [
            {"step": "1.列名统一", "detail": "target 重命名为 MEDV"},
            {"step": "2.缺失检查", "detail": f"缺失值总数={int(raw.isna().sum().sum())}"},
            {"step": "3.封顶值记录", "detail": f"MEDV>=50 的样本 {int((raw['MEDV']>=50).sum())} 条，默认保留（真实审查上限）"},
            {"step": "4.训练测试划分", "detail": f"test_size=0.2, n_train={len(bundle.y_train)}, n_test={len(bundle.y_test)}"},
            {"step": "5.标准化", "detail": "树模型不依赖缩放；标准化结果一并保存以便对照"},
        ]
    )
    save_csv(steps, RESULT_DIR / "boston_preprocess_steps.csv")

    feat = bundle.feature_names
    show = feat[:6]
    fig, axes = plt.subplots(2, 6, figsize=(15.5, 5.6))
    for j, col in enumerate(show):
        idx = feat.index(col)
        axes[0, j].hist(bundle.X_train[:, idx], bins=16, color=COLORS["teal"], edgecolor="white")
        axes[0, j].set_title(cn(col), fontsize=9, fontweight="bold")
        axes[1, j].hist(bundle.X_train_scaled[:, idx], bins=16, color=COLORS["cyan"], edgecolor="white")
    axes[0, 0].set_ylabel("原始")
    axes[1, 0].set_ylabel("标准化后")
    fig.suptitle("Boston 预处理：标准化前后分布", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out / "scale_before_after.png")

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.hist(bundle.y_train, bins=18, alpha=0.85, color=COLORS["teal"], label="训练", edgecolor="white")
    ax.hist(bundle.y_test, bins=18, alpha=0.75, color=COLORS["coral"], label="测试", edgecolor="white")
    ax.set_xlabel("MEDV")
    ax.set_title("划分后房价分布", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, out / "split_y_hist.png")

    info = {
        "n_train": int(len(bundle.y_train)),
        "n_test": int(len(bundle.y_test)),
        "n_capped50": int((raw["MEDV"] >= 50).sum()),
        "y_mean_train": float(bundle.y_train.mean()),
        "y_mean_test": float(bundle.y_test.mean()),
    }
    save_json(info, RESULT_DIR / "boston_preprocess_summary.json")
    print("  Boston 预处理完成")
    return info


if __name__ == "__main__":
    run()
