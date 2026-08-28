from __future__ import annotations

import inspect

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from titanic_ml.config.settings import RANDOM_STATE


def make_adaboost():
    stump = DecisionTreeClassifier(
        max_depth=2, criterion="entropy", class_weight="balanced", random_state=RANDOM_STATE
    )
    kwargs = {"random_state": RANDOM_STATE}
    sig = inspect.signature(AdaBoostClassifier.__init__)
    if "estimator" in sig.parameters:
        kwargs["estimator"] = stump
        depth_key = "estimator__max_depth"
    else:
        kwargs["base_estimator"] = stump
        depth_key = "base_estimator__max_depth"
    return AdaBoostClassifier(**kwargs), {
        "n_estimators": [60, 100, 140],
        "learning_rate": [0.03, 0.1, 0.3],
        depth_key: [1, 2, 3],
    }


def build_model_zoo():
    ada, ada_grid = make_adaboost()
    return {
        "DecisionTree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            {
                "max_depth": [3, 5, 7, None],
                "min_samples_split": [2, 8, 16],
                "min_samples_leaf": [1, 4, 8],
                "criterion": ["gini", "entropy"],
            },
        ),
        "LogisticRegression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000, solver="liblinear", random_state=RANDOM_STATE
                        ),
                    ),
                ]
            ),
            {
                "clf__C": [0.1, 1.0, 3.0, 10.0],
                "clf__penalty": ["l1", "l2"],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "AdaBoost": (ada, ada_grid),
        "RandomForest": (
            RandomForestClassifier(
                min_samples_leaf=4,
                min_samples_split=8,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {"n_estimators": [80, 160, 240], "max_depth": [4, 6, 8]},
        ),
        "GBDT": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [80, 140, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
                "subsample": [0.8, 1.0],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
            ),
            {
                "n_estimators": [80, 140, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 4],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        ),
    }
