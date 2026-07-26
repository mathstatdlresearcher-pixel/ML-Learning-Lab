"""可视化主题。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

COLORS = {
    "bg": "#F4F7FA",
    "panel": "#FFFFFF",
    "ink": "#1E293B",
    "muted": "#64748B",
    "grid": "#CBD5E1",
    "teal": "#0F766E",
    "cyan": "#0891B2",
    "coral": "#E11D48",
    "amber": "#D97706",
    "slate": "#334155",
    "mint": "#14B8A6",
    "series": ["#0F766E", "#E11D48", "#D97706", "#0891B2", "#4F46E5", "#059669"],
}


def _pick_cjk_font() -> Optional[str]:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def apply_theme() -> None:
    cjk = _pick_cjk_font()
    font_list = [cjk, "Segoe UI", "DejaVu Sans"] if cjk else ["Segoe UI", "DejaVu Sans"]
    mpl.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["panel"],
            "savefig.facecolor": COLORS["bg"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["ink"],
            "grid.color": COLORS["grid"],
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
            "legend.frameon": False,
            "font.family": "sans-serif",
            "font.sans-serif": font_list,
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "lines.linewidth": 2.2,
        }
    )


def save_fig(fig: plt.Figure, path: Path, close: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    if close:
        plt.close(fig)
    return path
