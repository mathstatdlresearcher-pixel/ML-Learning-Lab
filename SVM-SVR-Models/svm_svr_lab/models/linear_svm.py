"""线性 SVM 分类（模拟数据）。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm

from common.config import FIG_DIR, RESULT_DIR
from common.utils import clf_metrics, save_fig, setup_font
from data_loader.prepare_data import load_blobs_data


def run_linear_svm(C=1.0):
    setup_font()
    data = load_blobs_data()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    clf = svm.SVC(C=C, kernel="linear")
    clf.fit(X_train, y_train.ravel())

    train_m = clf_metrics(y_train, clf.predict(X_train))
    test_m = clf_metrics(y_test, clf.predict(X_test))
    w1, w2 = clf.coef_[0]
    b = float(clf.intercept_[0])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolors="k")
    xs = np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 200)
    ax.plot(xs, -w1 / w2 * xs - b / w2, "k-", lw=2, label="决策边界")
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=120,
        facecolors="none",
        edgecolors="green",
        linewidths=1.5,
        label="支持向量",
    )
    ax.set_title(f"线性 SVM (C={C})")
    ax.legend()
    save_fig(fig, FIG_DIR / "linear_svm" / "decision_boundary.png")

    row = {
        "C": C,
        "w1": float(w1),
        "w2": float(w2),
        "b": b,
        "n_support": int(clf.support_vectors_.shape[0]),
        **{f"train_{k}": v for k, v in train_m.items()},
        **{f"test_{k}": v for k, v in test_m.items()},
    }
    df = pd.DataFrame([row])
    df.to_csv(RESULT_DIR / "linear_svm_metrics.csv", index=False)
    print(df.to_string(index=False))
    return row


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    run_linear_svm()