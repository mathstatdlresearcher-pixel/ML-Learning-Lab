from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def setup_font():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def save_fig(fig, path, dpi=150):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存图片: {path}")
    return path


def clf_metrics(y_true, y_pred):
    return {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def reg_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "MAE": float(metrics.mean_absolute_error(y_true, y_pred)),
        "MSE": float(metrics.mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(metrics.mean_squared_error(y_true, y_pred))),
        "MAPE": float(metrics.mean_absolute_percentage_error(y_true, y_pred)),
        "R2": float(metrics.r2_score(y_true, y_pred)),
    }


def print_reg(y_true, y_pred, title=None):
    m = reg_metrics(y_true, y_pred)
    if title:
        print(f"**********{title}***********")
    for k, v in m.items():
        print(f"{k}: {v}")
    print("__________________________________")
    return m