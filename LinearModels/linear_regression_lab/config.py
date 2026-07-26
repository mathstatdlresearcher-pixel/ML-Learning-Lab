"""全局配置。"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"
DATA_DIR = OUTPUT_DIR / "data"

for _d in (OUTPUT_DIR, FIG_DIR, REPORT_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SAMPLES = 80
TEST_SIZE = 0.3
N_REPEATS = 25

FEATURE_SCALE = 1.0

# 无关特征更多 + 样本更少 → OLS 过拟合，Lasso/LARS 优势更明显
N_JUNK = 16

NOISE_RATIO = {
    "y1": 0.0,
    "y2": 0.25,
    "y3": 0.15,
    "y4": 0.28,
}

RIDGE_ALPHAS = np.logspace(-3, 3, 30)
LASSO_ALPHAS = np.logspace(-4, 1, 30)

BOSTON_FEATURE_NAMES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
]
