"""依次运行全部实验。"""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import runtime  # noqa: F401  须先限制 OpenMP 线程

SCRIPTS = [
    ROOT / "experiments" / "01_iris_eda.py",
    ROOT / "experiments" / "02_iris_four_algorithms.py",
    ROOT / "experiments" / "03_kmeans_k_parameter.py",
    ROOT / "experiments" / "04_synthetic_params.py",
    ROOT / "experiments" / "05_write_report.py",
]


def main():
    for script in SCRIPTS:
        print("\n" + "=" * 60)
        print(f"运行 {script.name}")
        print("=" * 60)
        runpy.run_path(str(script), run_name="__main__")
    print("\n全部实验结束。图表: outputs/figures/  表格: outputs/tables/  报告: 实验报告.md")


if __name__ == "__main__":
    main()
