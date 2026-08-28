from __future__ import annotations

import numpy as np
import pandas as pd

from titanic_ml.data.features import lagrange_fill


def drop_high_missing(train: pd.DataFrame, others: list[pd.DataFrame], thresh: float = 0.5):
    miss_ratio = train.isnull().mean()
    drop_cols = miss_ratio[miss_ratio > thresh].index.tolist()
    train_k = train.drop(columns=[c for c in drop_cols if c in train.columns])
    others_k = [df.drop(columns=[c for c in drop_cols if c in df.columns]) for df in others]
    return drop_cols, miss_ratio, train_k, others_k


def fill_missing(train: pd.DataFrame, others: list[pd.DataFrame]):
    train = train.copy()
    others = [df.copy() for df in others]
    all_dfs = [train] + others
    text_cols = train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in train.select_dtypes(include=[np.number]).columns if c != "Survived"]
    fill_log = {}
    for col in text_cols:
        any_miss = any(col in df.columns and df[col].isnull().any() for df in all_dfs)
        if not any_miss:
            continue
        mode_val = train[col].mode(dropna=True)
        mode_val = mode_val.iloc[0] if len(mode_val) else "Unknown"
        fill_log[col] = {"type": "mode", "value": str(mode_val)}
        for df in all_dfs:
            if col in df.columns:
                df[col] = df[col].fillna(mode_val)
    clips = {"Age": (0.0, 80.0), "Fare": (0.0, 600.0)}
    for col in num_cols:
        any_miss = any(col in df.columns and df[col].isnull().any() for df in all_dfs)
        if not any_miss:
            continue
        fill_log[col] = {"type": "lagrange", "k": 5}
        clip = clips.get(col)
        train[col] = lagrange_fill(train[col], clip=clip)
        med = train[col].median()
        for i, df in enumerate(others):
            df[col] = lagrange_fill(df[col], clip=clip).fillna(med)
            others[i] = df
        all_dfs = [train] + others
    return train, others, fill_log


def bin_and_dummy(train: pd.DataFrame, others: list[pd.DataFrame]):
    train = train.copy()
    others = [df.copy() for df in others]
    all_dfs = [train] + others
    age_bins = [0, 12, 18, 35, 50, 80]
    age_labels = ["Child", "Teen", "YoungAdult", "MidAge", "Senior"]
    for df in all_dfs:
        df["AgeBin"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, include_lowest=True)
        df["IsMother"] = ((df["Title"] == "Mrs") & (df["Parch"] > 0)).astype(int)
        df["FareLog"] = np.log1p(df["Fare"])
    train, others = all_dfs[0], all_dfs[1:]
    fare_bins = pd.qcut(train["Fare"], q=4, retbins=True, duplicates="drop")[1]
    fare_bins[0] = min(fare_bins[0], 0)
    fare_max = max(df["Fare"].max() for df in [train] + others)
    fare_bins[-1] = max(fare_bins[-1], float(fare_max))
    fare_labels = [f"Q{i + 1}" for i in range(len(fare_bins) - 1)]
    for df in [train] + others:
        df["FareBin"] = pd.cut(df["Fare"], bins=fare_bins, labels=fare_labels, include_lowest=True)

    dummy_cols = [
        "Sex", "Embarked", "Title", "Pclass", "AgeBin", "FareBin",
        "TicketGroup", "FamilyBin", "CabinDeck", "SexPclass",
    ]
    dummy_cols = [c for c in dummy_cols if c in train.columns]
    train_d = pd.get_dummies(train, columns=dummy_cols, drop_first=False)
    other_d = [pd.get_dummies(df, columns=dummy_cols, drop_first=False) for df in others]
    aligned = []
    for d in other_d:
        train_d, d = train_d.align(d, join="left", axis=1, fill_value=0)
        aligned.append(d)

    drop_always = ["Name", "Ticket", "Cabin", "PassengerId"]
    feature_cols = [
        c for c in train_d.columns
        if c not in drop_always + ["Survived"] and train_d[c].dtype != "object"
    ]
    X_train = train_d[feature_cols].astype(float)
    X_others = [d[feature_cols].astype(float) for d in aligned]
    y_train = train["Survived"].astype(int)
    y_others = [df["Survived"].astype(int) if "Survived" in df.columns else None for df in others]
    return X_train, X_others, y_train, y_others, fare_bins, dummy_cols, feature_cols
