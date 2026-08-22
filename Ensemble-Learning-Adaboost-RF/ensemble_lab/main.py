"""
集成学习实验入口（AdaBoost / Random Forest / Lars）

用法:
  D:\\Anaconda3\\envs\\Pytorch\\python.exe main.py
  D:\\Anaconda3\\envs\\Pytorch\\python.exe main.py --quick
  D:\\Anaconda3\\envs\\Pytorch\\python.exe main.py --only 01,05
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.style import apply_theme

EXPERIMENTS = [
    ("01", "Breast Cancer 探索性分析", "experiments.01_breast_eda"),
    ("02", "Breast Cancer 数据预处理", "experiments.02_breast_preprocess"),
    ("03", "AdaBoost 不同基学习器网格搜索", "experiments.03_breast_adaboost"),
    ("04", "AdaBoost / RF / Lars 关键特征", "experiments.04_breast_features"),
    ("05", "Boston 探索性分析", "experiments.05_boston_eda"),
    ("06", "Boston 数据预处理", "experiments.06_boston_preprocess"),
    ("07", "Random Forest 参数影响", "experiments.07_boston_rf_params"),
    ("08", "单决策树 vs AdaBoost vs RF", "experiments.08_boston_ensemble"),
]


def _import_run(module_path: str):
    import importlib

    return importlib.import_module(module_path).run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="集成学习 AdaBoost / RF 全流程实验")
    parser.add_argument("--only", type=str, default="", help="只运行指定编号，如 01,03")
    parser.add_argument("--quick", action="store_true", help="缩小网格，便于快速试跑")
    args = parser.parse_args(argv)

    apply_theme()
    selected = {x.strip().zfill(2) for x in args.only.split(",") if x.strip()}
    jobs = [e for e in EXPERIMENTS if not selected or e[0] in selected]

    print("\n" + "=" * 64)
    print("  Ensemble Lab | AdaBoost / Random Forest / Lars")
    print("=" * 64)
    print(f"Python env: {sys.executable}")
    print(f"Running {len(jobs)} module(s), quick={args.quick}\n")

    t0 = time.time()
    failed = []
    for code, title, module_path in jobs:
        print(f">> [{code}] {title}")
        try:
            run = _import_run(module_path)
            try:
                run(quick=args.quick)
            except TypeError:
                run()
        except Exception:  # noqa: BLE001
            failed.append(code)
            traceback.print_exc()
            print(f"XX [{code}] FAILED\n")
            continue
        print(f"OK [{code}] done\n")

    elapsed = time.time() - t0
    print("=" * 64)
    if failed:
        print(f"Finished with failures: {failed}, elapsed {elapsed:.1f}s")
        return 1
    print(f"All done. Elapsed {elapsed:.1f}s")
    print(f"Figures: {ROOT / 'outputs' / 'figures'}")
    print(f"Results: {ROOT / 'outputs' / 'results'}")
    print("=" * 64 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
