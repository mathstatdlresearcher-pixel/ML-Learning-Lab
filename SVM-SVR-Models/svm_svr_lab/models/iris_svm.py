"""鸢尾花 SVM：核函数对比 + RBF/Poly 参数敏感性 + 网格搜索。"""

from __future__ import annotations

import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.svm import SVC

from common.config import FIG_DIR, IRIS_FEATURES, RANDOM_STATE, RESULT_DIR, TEST_SIZE
from common.utils import clf_metrics, save_fig, setup_font
from data_loader.prepare_data import load_iris_data


def _plot_score_time(labels, train_scores, test_scores, times, title, path):
    x = list(range(len(labels)))
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax1.plot(x, train_scores, "r-", lw=2, label="训练集准确率")
    ax1.plot(x, test_scores, "g-", lw=2, label="测试集准确率")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.legend(loc="lower left")
    ax1.set_title("准确率")
    ax1.grid(True, ls=":")

    ax2 = fig.add_subplot(122)
    ax2.plot(x, times, "b-", lw=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_title("训练耗时(秒)")
    ax2.grid(True, ls=":")
    fig.suptitle(title)
    fig.tight_layout()
    save_fig(fig, path)


def _plot_boundary(models, titles, X, y, X_test, feat_names, path):
    N = 400
    x1_min, x2_min = X.min(axis=0)
    x1_max, x2_max = X.max(axis=0)
    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, N), np.linspace(x2_min, x2_max, N)
    )
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    cm_light = mpl.colors.ListedColormap(["#A0FFA0", "#FFA0A0", "#A0A0FF"])
    cm_dark = mpl.colors.ListedColormap(["g", "r", "b"])

    fig = plt.figure(figsize=(12, 10))
    for i, (model, title) in enumerate(zip(models, titles), 1):
        ax = fig.add_subplot(2, 2, i)
        hat = model.predict(grid).reshape(xx1.shape)
        ax.pcolormesh(xx1, xx2, hat, cmap=cm_light, shading="auto")
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=40, cmap=cm_dark)
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            s=100,
            facecolors="none",
            edgecolors="k",
            zorder=10,
        )
        ax.set_xlabel(feat_names[0])
        ax.set_ylabel(feat_names[1])
        ax.set_title(title)
        ax.grid(True, ls=":")
    fig.tight_layout()
    save_fig(fig, path)


def compare_kernels(X, y, feature_pair, tag):
    cols = list(feature_pair)
    X_pair = X.iloc[:, cols].values
    feat_names = [IRIS_FEATURES[i] for i in cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X_pair, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    kernels = ["linear", "rbf", "poly", "sigmoid"]
    models, trs, tes, times, rows = [], [], [], [], []
    for k in kernels:
        model = SVC(kernel=k)
        t0 = time.time()
        model.fit(X_train, y_train)
        dt = time.time() - t0
        tr = accuracy_score(y_train, model.predict(X_train))
        te = accuracy_score(y_test, model.predict(X_test))
        models.append(model)
        trs.append(tr)
        tes.append(te)
        times.append(dt)
        m = clf_metrics(y_test, model.predict(X_test))
        rows.append(
            {
                "experiment": tag,
                "kernel": k,
                "features": ",".join(feat_names),
                "train_accuracy": tr,
                "test_accuracy": te,
                "fit_time_sec": dt,
                **{f"test_{kk}": vv for kk, vv in m.items() if kk != "accuracy"},
            }
        )

    out = FIG_DIR / "iris_svm"
    _plot_score_time(
        [f"{k}-SVM" for k in kernels],
        trs,
        tes,
        times,
        f"核函数对比 ({'/'.join(feat_names)})",
        out / f"kernel_compare_{tag}.png",
    )
    if X_pair.shape[1] == 2:
        _plot_boundary(
            models,
            [f"{k}-SVM" for k in kernels],
            X_pair,
            y,
            X_test,
            feat_names,
            out / f"decision_boundary_{tag}.png",
        )
    return pd.DataFrame(rows)


def rbf_sensitivity(X, y):
    X2 = X.iloc[:, [0, 1]].values
    X_train, X_test, y_train, y_test = train_test_split(
        X2, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    out = FIG_DIR / "iris_svm"
    rows = []

    def _run(param_name, values, builder, title, score_name, bound_name):
        models, titles, trs, tes, times = [], [], [], [], []
        for v in values:
            model = builder(v)
            t0 = time.time()
            model.fit(X_train, y_train)
            dt = time.time() - t0
            tr = accuracy_score(y_train, model.predict(X_train))
            te = accuracy_score(y_test, model.predict(X_test))
            models.append(model)
            titles.append(f"RBF {param_name}={v}")
            trs.append(tr)
            tes.append(te)
            times.append(dt)
            rows.append({"param": param_name, "value": v, "train_acc": tr, "test_acc": te})
        _plot_score_time(titles, trs, tes, times, title, out / score_name)
        _plot_boundary(models, titles, X2, y, X_test, IRIS_FEATURES[:2], out / bound_name)

    _run(
        "C",
        [0.3, 1, 10, 1000],
        lambda c: SVC(C=c, kernel="rbf", gamma=1),
        "RBF: C 敏感性",
        "rbf_C_sensitivity.png",
        "rbf_C_boundary.png",
    )
    _run(
        "gamma",
        [0.1, 1, 10, 100],
        lambda g: SVC(C=0.3, kernel="rbf", gamma=g),
        "RBF: gamma 敏感性",
        "rbf_gamma_sensitivity.png",
        "rbf_gamma_boundary.png",
    )
    return pd.DataFrame(rows)


def poly_sensitivity(X, y):
    X2 = X.iloc[:, [0, 1]].values
    X_train, X_test, y_train, y_test = train_test_split(
        X2, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    out = FIG_DIR / "iris_svm"
    rows = []

    def _run(param_name, values, builder, title, score_name, bound_name):
        models, titles, trs, tes, times = [], [], [], [], []
        for v in values:
            model = builder(v)
            t0 = time.time()
            model.fit(X_train, y_train)
            dt = time.time() - t0
            tr = accuracy_score(y_train, model.predict(X_train))
            te = accuracy_score(y_test, model.predict(X_test))
            models.append(model)
            titles.append(f"poly {param_name}={v}")
            trs.append(tr)
            tes.append(te)
            times.append(dt)
            rows.append({"param": param_name, "value": v, "train_acc": tr, "test_acc": te})
        _plot_score_time(titles, trs, tes, times, title, out / score_name)
        _plot_boundary(models, titles, X2, y, X_test, IRIS_FEATURES[:2], out / bound_name)

    _run(
        "degree",
        [1, 2, 5, 10],
        lambda d: SVC(C=0.5, kernel="poly", degree=d),
        "Poly: degree 敏感性",
        "poly_degree_sensitivity.png",
        "poly_degree_boundary.png",
    )
    _run(
        "C",
        [0.1, 0.5, 10, 1000],
        lambda c: SVC(C=c, kernel="poly", degree=2),
        "Poly: C 敏感性",
        "poly_C_sensitivity.png",
        "poly_C_boundary.png",
    )
    return pd.DataFrame(rows)


def grid_search_rbf(X, y):
    X2 = X.iloc[:, [0, 1]].values
    X_train, X_test, y_train, y_test = train_test_split(
        X2, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    param_grid = {
        "C": [0.1, 0.3, 0.5, 1, 3, 5, 10],
        "gamma": [0.1, 0.5, 1, 3, 5, 10, 13],
    }
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=12)
    grid = GridSearchCV(
        SVC(kernel="rbf"), param_grid=param_grid, scoring="accuracy", cv=cv, n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    result = {
        "best_params": str(grid.best_params_),
        "best_cv_score": float(grid.best_score_),
        "train_accuracy": float(accuracy_score(y_train, best.predict(X_train))),
        "test_accuracy": float(accuracy_score(y_test, best.predict(X_test))),
    }
    pd.DataFrame([result]).to_csv(RESULT_DIR / "iris_rbf_gridsearch.csv", index=False)
    print("RBF GridSearch:", result)
    return result


def run_iris_svm(do_grid_search=True):
    setup_font()
    data = load_iris_data()
    X, y = data["X"], data["y"]

    kernel_df = pd.concat(
        [
            compare_kernels(X, y, (2, 3), "petal"),
            compare_kernels(X, y, (0, 1), "sepal"),
            compare_kernels(X, y, (0, 1, 2, 3), "all4"),
        ],
        ignore_index=True,
    )
    rbf_df = rbf_sensitivity(X, y)
    poly_df = poly_sensitivity(X, y)

    kernel_df.to_csv(RESULT_DIR / "iris_kernel_compare.csv", index=False)
    rbf_df.to_csv(RESULT_DIR / "iris_rbf_sensitivity.csv", index=False)
    poly_df.to_csv(RESULT_DIR / "iris_poly_sensitivity.csv", index=False)

    if do_grid_search:
        grid_search_rbf(X, y)

    print(kernel_df.to_string(index=False))
    return kernel_df


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-grid", action="store_true")
    args = parser.parse_args()
    run_iris_svm(do_grid_search=not args.no_grid)