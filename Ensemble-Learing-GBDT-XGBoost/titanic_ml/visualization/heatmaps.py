from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _cell_text(val: float, fmt: str) -> str:
    if fmt == "d":
        return f"{int(round(val))}"
    if fmt.startswith("."):
        return format(float(val), fmt)
    return format(float(val), fmt)


def annotated_heatmap(
    data: pd.DataFrame,
    ax,
    *,
    fmt: str = ".3f",
    cmap: str = "YlGnBu",
    vmin=None,
    vmax=None,
    center=None,
    cbar: bool = True,
    annot_size: int = 11,
    x_rotation: int = 20,
) -> None:
    """Heatmap with a number in every cell and horizontal y-axis labels."""
    frame = pd.DataFrame(data)
    arr = frame.to_numpy(dtype=float)
    heat_kw = dict(
        annot=False,
        cmap=cmap,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cbar=cbar,
        linewidths=0.7,
        linecolor="white",
    )
    if cbar:
        heat_kw["cbar_kws"] = {"shrink": 0.82}
    sns.heatmap(frame, **heat_kw)
    cmap_obj = plt.get_cmap(cmap)
    lo = np.nanmin(arr) if vmin is None else float(vmin)
    hi = np.nanmax(arr) if vmax is None else float(vmax)
    if center is not None:
        span = max(abs(lo - center), abs(hi - center)) or 1.0
        lo, hi = center - span, center + span
    denom = (hi - lo) or 1.0

    n_row, n_col = arr.shape
    for i in range(n_row):
        for j in range(n_col):
            val = arr[i, j]
            if not np.isfinite(val):
                continue
            norm = (val - lo) / denom
            if center is not None:
                r, g, b, _ = cmap_obj(0.5 + 0.5 * np.clip((val - center) / (span or 1), -1, 1))
            else:
                r, g, b, _ = cmap_obj(np.clip(norm, 0, 1))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            color = "white" if luminance < 0.55 else "#111111"
            ax.text(
                j + 0.5,
                i + 0.5,
                _cell_text(val, fmt),
                ha="center",
                va="center",
                fontsize=annot_size,
                color=color,
                clip_on=False,
                zorder=5,
            )

    ax.set_xticks(np.arange(n_col) + 0.5)
    ax.set_xticklabels([str(c) for c in frame.columns], rotation=x_rotation, ha="right")
    ax.set_yticks(np.arange(n_row) + 0.5)
    ax.set_yticklabels([str(i) for i in frame.index], rotation=0, ha="right", va="center")
    ax.tick_params(axis="y", labelrotation=0, pad=8, labelsize=11, length=0)
    ax.tick_params(axis="x", labelsize=10, length=0)
