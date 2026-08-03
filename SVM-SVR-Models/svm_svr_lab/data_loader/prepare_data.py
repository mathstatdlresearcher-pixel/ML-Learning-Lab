"""数据集：全部在代码中下载 / 生成 / 划分。"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_openml, load_iris, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common.config import IRIS_CLASSES, IRIS_FEATURES, RANDOM_STATE, TEST_SIZE


def load_iris_data(test_size=TEST_SIZE, random_state=RANDOM_STATE):
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = IRIS_FEATURES + ["class_id"]
    df["class"] = df["class_id"].map(dict(enumerate(IRIS_CLASSES)))

    X = df[IRIS_FEATURES]
    y = df["class_id"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Iris] n={len(df)}, train={len(X_train)}, test={len(X_test)}")
    return {
        "full": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": IRIS_FEATURES,
        "class_names": IRIS_CLASSES,
    }


def load_abalone_data(test_size=0.2, random_state=42, scale=True):
    """从 OpenML 下载鲍鱼数据，Sex one-hot，划分训练/测试。"""
    ds = fetch_openml(name="abalone", version=1, as_frame=True, parser="auto")
    df = ds.frame.copy()

    target_candidates = ["age", "Rings", "Class_number_of_rings"]
    target_col = next(c for c in target_candidates if c in df.columns)
    df = df.rename(columns={target_col: "age"})
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    rename = {
        "Whole_weight": "Whole weight",
        "Shucked_weight": "Shucked weight",
        "Viscera_weight": "Viscera weight",
        "Shell_weight": "Shell weight",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "Sex" in df.columns:
        dummies = pd.get_dummies(df["Sex"], prefix="Sex")
        for col in ["Sex_F", "Sex_M"]:
            if col not in dummies.columns:
                dummies[col] = 0
        df = pd.concat([df.drop(columns=["Sex"]), dummies[["Sex_F", "Sex_M"]]], axis=1)

    df = df.dropna(subset=["age"]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != "age"]
    X = df[feature_cols].astype(float)
    y = df["age"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test), columns=feature_cols, index=X_test.index
        )

    print(
        f"[Abalone] n={len(df)}, features={len(feature_cols)}, "
        f"train={len(X_train)}, test={len(X_test)}"
    )
    return {
        "full": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_cols,
    }


def load_blobs_data(
    n_samples=200,
    cluster_std=1.2,
    random_state=40,
    test_size=0.3,
):
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=2,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    y = y.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=43
    )
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y
    print(f"[Blobs] n={len(df)}, train={len(X_train)}, test={len(X_test)}")
    return {
        "full": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    load_iris_data()
    load_blobs_data()
    load_abalone_data()