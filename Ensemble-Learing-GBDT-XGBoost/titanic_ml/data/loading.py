from __future__ import annotations

import pandas as pd

from titanic_ml.config.settings import DATA_DIR


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test
