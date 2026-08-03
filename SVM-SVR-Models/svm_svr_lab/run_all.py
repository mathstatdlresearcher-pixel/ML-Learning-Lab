"""SVM-SVR-Models 一键运行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader.prepare_data import load_abalone_data, load_blobs_data, load_iris_data
from eda.eda import run_eda
from models.iris_svm import run_iris_svm
from models.linear_svm import run_linear_svm
from models.svr import run_svr


def main():
    parser = argparse.ArgumentParser(description="SVM / SVR 全流程实验")
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-linear", action="store_true")
    parser.add_argument("--skip-iris", action="store_true")
    parser.add_argument("--skip-svr", action="store_true")
    parser.add_argument("--quick-svr", action="store_true")
    parser.add_argument("--no-grid", action="store_true")
    args = parser.parse_args()

    print("=" * 50, "\n1) 数据下载/生成/划分\n", "=" * 50)
    load_iris_data()
    load_blobs_data()
    load_abalone_data()

    if not args.skip_eda:
        print("=" * 50, "\n2) EDA\n", "=" * 50)
        run_eda()

    if not args.skip_linear:
        print("=" * 50, "\n3) 线性 SVM\n", "=" * 50)
        run_linear_svm()

    if not args.skip_iris:
        print("=" * 50, "\n4) 鸢尾花 SVM\n", "=" * 50)
        run_iris_svm(do_grid_search=not args.no_grid)

    if not args.skip_svr:
        print("=" * 50, "\n5) SVR\n", "=" * 50)
        run_svr(quick=args.quick_svr)

    print("\n完成。图: outputs/figures  指标: outputs/results")


if __name__ == "__main__":
    main()