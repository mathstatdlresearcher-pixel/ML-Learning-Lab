from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "titanic-dataset"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 2023
MODEL_ORDER = [
    "DecisionTree",
    "LogisticRegression",
    "AdaBoost",
    "RandomForest",
    "GBDT",
    "XGBoost",
]
