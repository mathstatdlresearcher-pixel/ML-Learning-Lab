from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
RESULT_DIR = ROOT / "outputs" / "results"

IRIS_FEATURES = ["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"]
IRIS_CLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

RANDOM_STATE = 28
TEST_SIZE = 0.4

for p in (FIG_DIR, RESULT_DIR):
    p.mkdir(parents=True, exist_ok=True)