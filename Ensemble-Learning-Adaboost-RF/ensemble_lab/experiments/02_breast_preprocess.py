"""二.2 Breast Cancer 数据预处理。"""

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
from data_loader.prepare_data import load_breast_raw, prepare_breast


def run() -> dict:
    apply_theme()
    out = FIG_DIR / "02_breast_preprocess"
    out.mkdir(parents=True, exist_ok=True)

    raw = load_breast_raw()
    bundle = prepare_breast()

    feat = bundle.feature_names
    steps = pd.DataFrame(
        [
            {"step": "1.列名去空格", "detail": "concave points_worst 等列名去首尾空格"},
            {"step": "2.标签重编码", "detail": "1=恶性, 0=良性，便于 Precision/Recall 关注阳性类"},
            {"step": "3.缺失检查", "detail": f"缺失值总数={int(raw.isna().sum().sum())}，无需填补"},
            {"step": "4.重复检查", "detail": f"重复行={int(raw.duplicated().sum())}"},
            {"step": "5.IQR 截尾", "detail": "1.5→3.0 IQR 温和截尾，降低极端值对 LR/SVM 的影响"},
            {"step": "6.分层划分", "detail": f"test_size=0.2, stratify, n_train={len(bundle.y_train)}, n_test={len(bundle.y_test)}"},
            {"step": "7.标准化", "detail": "StandardScaler 仅拟合训练集，供 LR / SVM / Lars 使用；树模型用原始尺度"},
        ]
    )
    save_csv(steps, RESULT_DIR / "breast_preprocess_steps.csv")

    # 标准化前后对比（取 6 个特征）
    show = feat[:6]
    fig, axes = plt.subplots(2, 6, figsize=(15.5, 5.6))
    for j, col in enumerate(show):
        idx = feat.index(col)
        axes[0, j].hist(bundle.X_train[:, idx], bins=18, color=COLORS["teal"], edgecolor="white")
        axes[0, j].set_title(cn(col), fontsize=9, fontweight="bold")
        axes[1, j].hist(bundle.X_train_scaled[:, idx], bins=18, color=COLORS["cyan"], edgecolor="white")
    axes[0, 0].set_ylabel("原始")
    axes[1, 0].set_ylabel("标准化后")
    fig.suptitle("预处理：标准化前后特征分布", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out / "scale_before_after.png")

    # 训练/测试标签比例
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    labels = ["训练集", "测试集"]
    mal = [bundle.y_train.mean(), bundle.y_test.mean()]
    ben = [1 - m for m in mal]
    ax.bar(labels, ben, color=COLORS["teal"], label="良性", edgecolor="white")
    ax.bar(labels, mal, bottom=ben, color=COLORS["coral"], label="恶性", edgecolor="white")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("比例")
    ax.set_title("分层抽样后的类别比例", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, out / "split_class_ratio.png")

    info = {
        "n_features": len(feat),
        "n_train": int(len(bundle.y_train)),
        "n_test": int(len(bundle.y_test)),
        "malignant_train": float(bundle.y_train.mean()),
        "malignant_test": float(bundle.y_test.mean()),
        "scaled_mean_abs": float(np.abs(bundle.X_train_scaled.mean(axis=0)).mean()),
        "scaled_std_mean": float(bundle.X_train_scaled.std(axis=0).mean()),
    }
    save_json(info, RESULT_DIR / "breast_preprocess_summary.json")
    print("  Breast 预处理完成")
    return info


if __name__ == "__main__":
    run()
