"""SVR 回归：鲍鱼年龄预测。"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

from common.config import FIG_DIR, RESULT_DIR
from common.utils import print_reg, save_fig, setup_font
from data_loader.prepare_data import load_abalone_data


def default_kernel_compare(X_train, y_train, X_test, y_test):
    rows = []
    for kernel in ["linear", "poly", "rbf"]:
        model = SVR(kernel=kernel)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        m = print_reg(y_test, pred, title=kernel)
        rows.append({"kernel": kernel, "stage": "default", **m})
    return pd.DataFrame(rows)


def tune_gamma(X_train, y_train):
    gamma_exps = [0, 0.5, 1, 1.5, 2, 2.5]
    scores = []
    t0 = time.time()
    for g in gamma_exps:
        model = SVR(kernel="rbf", gamma=10**g)
        scores.append(float(cross_val_score(model, X_train, y_train, cv=5).mean()))
    elapsed = time.time() - t0
    best_i = int(np.argmax(scores))
    print(
        f"gamma=10^{gamma_exps[best_i]} 时 CV 最大={scores[best_i]:.6f}, 用时 {elapsed:.1f}s"
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gamma_exps, scores, marker="o")
    ax.set_xlabel("log10(gamma)")
    ax.set_ylabel("CV score")
    ax.set_title("SVR-RBF: gamma 调参")
    ax.grid(True, ls=":")
    save_fig(fig, FIG_DIR / "svr" / "rbf_gamma_cv.png")
    pd.DataFrame({"log10_gamma": gamma_exps, "cv_score": scores}).to_csv(
        RESULT_DIR / "svr_gamma_cv.csv", index=False
    )
    return 10 ** gamma_exps[best_i]


def tune_C(X_train, y_train):
    C_range = list(range(1, 51))
    scores = []
    t0 = time.time()
    for c in C_range:
        model = SVR(kernel="rbf", C=c)
        scores.append(float(cross_val_score(model, X_train, y_train, cv=5).mean()))
    elapsed = time.time() - t0
    best_i = int(np.argmax(scores))
    print(f"C={C_range[best_i]} 时 CV 最大={scores[best_i]:.6f}, 用时 {elapsed:.1f}s")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(C_range, scores, marker="o", markersize=3)
    ax.set_xlabel("C")
    ax.set_ylabel("CV score")
    ax.set_title("SVR-RBF: C 调参")
    ax.grid(True, ls=":")
    save_fig(fig, FIG_DIR / "svr" / "rbf_C_cv.png")
    pd.DataFrame({"C": C_range, "cv_score": scores}).to_csv(
        RESULT_DIR / "svr_C_cv.csv", index=False
    )
    return float(C_range[best_i])


def nested_search(X_train, y_train):
    C_range = list(range(1, 21))
    gamma_exps = [round(x, 1) for x in np.arange(0.1, 2.1, 0.2)]
    mat = np.zeros((len(C_range), len(gamma_exps)))
    t0 = time.time()
    for i, c in enumerate(C_range):
        for j, g in enumerate(gamma_exps):
            model = SVR(kernel="rbf", C=c, gamma=10**g)
            mat[i, j] = float(cross_val_score(model, X_train, y_train, cv=5).mean())
    elapsed = time.time() - t0
    idx = int(mat.argmax())
    i, j = idx // mat.shape[1], idx % mat.shape[1]
    print(
        f"网格最优 C={C_range[i]}, gamma=10^{gamma_exps[j]}, "
        f"CV={mat.max():.6f}, 用时 {elapsed:.1f}s"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(gamma_exps)))
    ax.set_xticklabels([str(g) for g in gamma_exps], rotation=45)
    ax.set_yticks(range(0, len(C_range), 2))
    ax.set_yticklabels([str(C_range[k]) for k in range(0, len(C_range), 2)])
    ax.set_xlabel("log10(gamma)")
    ax.set_ylabel("C")
    ax.set_title("SVR-RBF 网格搜索")
    fig.colorbar(im, ax=ax)
    save_fig(fig, FIG_DIR / "svr" / "rbf_grid_heatmap.png")

    records = [
        {"C": c, "log10_gamma": g, "cv_score": mat[ii, jj]}
        for ii, c in enumerate(C_range)
        for jj, g in enumerate(gamma_exps)
    ]
    pd.DataFrame(records).to_csv(RESULT_DIR / "svr_grid_cv.csv", index=False)
    return float(C_range[i]), float(10 ** gamma_exps[j])


def run_svr(quick=False):
    setup_font()
    data = load_abalone_data()
    X_train = data["X_train"].values
    X_test = data["X_test"].values
    y_train = data["y_train"].values.ravel()
    y_test = data["y_test"].values.ravel()

    print(">>> 默认核对比")
    default_df = default_kernel_compare(X_train, y_train, X_test, y_test)

    print(">>> 调 gamma")
    best_gamma = tune_gamma(X_train, y_train)
    print(">>> 调 C")
    best_c = tune_C(X_train, y_train)

    if quick:
        nest_c, nest_gamma = best_c, best_gamma
    else:
        print(">>> 网格搜索")
        nest_c, nest_gamma = nested_search(X_train, y_train)

    candidates = {
        "tuned_separate": (best_c, best_gamma),
        "tuned_nested": (nest_c, nest_gamma),
        "notebook_ref": (19, 10**0.5),
    }

    rows = default_df.to_dict("records")
    best_name, best_r2, best_pred = None, -np.inf, None
    for name, (c, g) in candidates.items():
        model = SVR(kernel="rbf", C=c, gamma=g)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        m = print_reg(y_test, pred, title=f"rbf/{name} C={c}, gamma={g}")
        rows.append({"kernel": "rbf", "stage": name, "C": c, "gamma": g, **m})
        if m["R2"] > best_r2:
            best_r2, best_name, best_pred = m["R2"], name, pred

    fig, ax = plt.subplots(figsize=(10, 4))
    idx = np.arange(len(y_test))
    ax.plot(idx, y_test, marker="o", markersize=2, label="真实值")
    ax.plot(idx, best_pred, marker="*", markersize=2, label="预测值")
    ax.legend()
    ax.set_title(f"SVR 预测对比 ({best_name})")
    save_fig(fig, FIG_DIR / "svr" / "pred_vs_true.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(best_pred, y_test - best_pred, alpha=0.5, s=15)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("预测值")
    ax.set_ylabel("残差")
    ax.set_title("残差图")
    save_fig(fig, FIG_DIR / "svr" / "residuals.png")

    result = pd.DataFrame(rows)
    result.to_csv(RESULT_DIR / "svr_model_compare.csv", index=False)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_svr(quick=args.quick)