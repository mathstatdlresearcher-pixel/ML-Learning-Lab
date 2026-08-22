"""全局路径、随机种子与特征中文名。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent

OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
RESULT_DIR = OUTPUT_DIR / "results"
DATA_OUT_DIR = OUTPUT_DIR / "data"
MODEL_DIR = OUTPUT_DIR / "models"

for _d in (OUTPUT_DIR, FIG_DIR, RESULT_DIR, DATA_OUT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_ROOT / "datasets"
BREAST_CSV = DATA_DIR / "breast_cancer.csv"
BOSTON_CSV = DATA_DIR / "boston.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV = 5
# Windows + 当前 sklearn/scipy 下 loky 多进程易崩，网格搜索改单进程
N_JOBS = 1

BOSTON_FEATURE_CN = {
    "CRIM": "人均犯罪率",
    "ZN": "大宅用地比例",
    "INDUS": "非零售商业用地比例",
    "CHAS": "是否临河",
    "NOX": "氮氧化物浓度",
    "RM": "平均房间数",
    "AGE": "老旧房屋比例",
    "DIS": "到就业中心距离",
    "RAD": "公路可达性",
    "TAX": "财产税率",
    "PTRATIO": "师生比",
    "B": "B指标",
    "LSTAT": "低收入人群比例",
    "target": "房价中位数",
    "MEDV": "房价中位数",
}

BREAST_FEATURE_CN = {
    "radius_mean": "半径-均值",
    "texture_mean": "纹理-均值",
    "perimeter_mean": "周长-均值",
    "area_mean": "面积-均值",
    "smoothness_mean": "平滑度-均值",
    "compactness_mean": "紧致度-均值",
    "concavity_mean": "凹度-均值",
    "concave points_mean": "凹点-均值",
    "symmetry_mean": "对称性-均值",
    "fractal_dimension_mean": "分形维-均值",
    "radius_se": "半径-标准误",
    "texture_se": "纹理-标准误",
    "perimeter_se": "周长-标准误",
    "area_se": "面积-标准误",
    "smoothness_se": "平滑度-标准误",
    "compactness_se": "紧致度-标准误",
    "concavity_se": "凹度-标准误",
    "concave points_se": "凹点-标准误",
    "symmetry_se": "对称性-标准误",
    "fractal_dimension_se": "分形维-标准误",
    "radius_worst": "半径-最值",
    "texture_worst": "纹理-最值",
    "perimeter_worst": "周长-最值",
    "area_worst": "面积-最值",
    "smoothness_worst": "平滑度-最值",
    "compactness_worst": "紧致度-最值",
    "concavity_worst": "凹度-最值",
    "concave points_worst": "凹点-最值",
    "symmetry_worst": "对称性-最值",
    "fractal_dimension_worst": "分形维-最值",
}


def cn(name: str) -> str:
    return BREAST_FEATURE_CN.get(name, BOSTON_FEATURE_CN.get(name, name))
