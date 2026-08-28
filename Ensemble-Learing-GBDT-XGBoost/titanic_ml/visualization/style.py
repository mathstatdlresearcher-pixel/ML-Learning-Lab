from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from titanic_ml.config.settings import FIG_DIR


def apply_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def savefig(name: str) -> None:
    fig = plt.gcf()
    for ax in fig.axes:
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right", va="center")
    fig.savefig(FIG_DIR / name, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def axes_grid(n: int, figsize=(14, 8)):
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def annotate_bars(ax, fmt="{:.3f}", fontsize: int = 7) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, fontsize=fontsize, padding=2, rotation=0)
