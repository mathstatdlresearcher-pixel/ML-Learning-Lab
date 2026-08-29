"""统一绘图风格。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"


def setup_style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def scatter_clusters(X, y, title, path, y_true=None, xlabel="x1", ylabel="x2"):
    setup_style()
    fig, axes = plt.subplots(1, 2 if y_true is not None else 1, figsize=(10 if y_true is not None else 5, 4))
    if y_true is None:
        axes = [axes]
    else:
        axes = list(np.atleast_1d(axes))
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", s=22, vmin=0, vmax=9, edgecolors="none")
    axes[0].set_title(title)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    if y_true is not None:
        axes[1].scatter(X[:, 0], X[:, 1], c=y_true, cmap="tab10", s=22, vmin=0, vmax=9, edgecolors="none")
        axes[1].set_title("真实标签")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def line_plot(xs, series, xlabel, ylabel, title, path, legend=None):
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, ys in series.items():
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend or len(series) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
