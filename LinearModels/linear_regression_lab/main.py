"""
线性回归算法实验 —— 一键运行入口

用法:
  python main.py              # 跑完全部实验
  python main.py --only 01    # 只跑 EDA
  python main.py --only 02,03 # 跑指定实验
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

from utils.style import apply_theme


EXPERIMENTS = [
    ("01", "模拟数据 EDA / 预处理", "experiments.01_simulated_eda"),
    ("02", "模拟数据模型对比", "experiments.02_model_comparison"),
    ("03", "LARS 变量选择稳定性", "experiments.03_lars_stability"),
    ("04", "Boston 探索性分析", "experiments.04_boston_eda"),
    ("05", "Boston 建模对比", "experiments.05_boston_modeling"),
]


def _import_run(module_path: str):
    import importlib

    mod = importlib.import_module(module_path)
    return mod.run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="线性回归算法实验流水线")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="只运行指定编号，逗号分隔，例如 01,02 或 4,5",
    )
    args = parser.parse_args(argv)

    apply_theme()
    selected = {x.strip().zfill(2) for x in args.only.split(",") if x.strip()}
    jobs = [e for e in EXPERIMENTS if not selected or e[0] in selected]

    print("\n" + "=" * 64)
    print("  Linear Regression Lab | OLS / Ridge / Lasso / LARS")
    print("=" * 64)
    print(f"Running {len(jobs)} experiment module(s)...\n")

    t0 = time.time()
    failed = []
    for code, title, module_path in jobs:
        print(f">> [{code}] {title}")
        try:
            run = _import_run(module_path)
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
        print(f"Figures: {ROOT / 'outputs' / 'figures'}")
        return 1
    print(f"All done. Elapsed {elapsed:.1f}s")
    print(f"Figures: {ROOT / 'outputs' / 'figures'}")
    print(f"Reports: {ROOT / 'outputs' / 'reports'}")
    print("=" * 64 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
