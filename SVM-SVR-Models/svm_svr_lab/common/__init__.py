"""公共配置与工具。"""

from .config import FIG_DIR, RESULT_DIR, ROOT
from .utils import clf_metrics, print_reg, reg_metrics, save_fig, setup_font

__all__ = [
    "ROOT",
    "FIG_DIR",
    "RESULT_DIR",
    "setup_font",
    "save_fig",
    "clf_metrics",
    "reg_metrics",
    "print_reg",
]