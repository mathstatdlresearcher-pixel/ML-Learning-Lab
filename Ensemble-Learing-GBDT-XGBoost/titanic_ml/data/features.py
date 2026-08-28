from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import lagrange


def lagrange_fill(series: pd.Series, k: int = 5, clip: tuple[float, float] | None = None) -> pd.Series:
    values = series.to_numpy(dtype=float).copy()
    n = len(values)
    missing = np.where(np.isnan(values))[0]
    for i in missing:
        idxs = np.concatenate(
            [np.arange(max(0, i - k), i), np.arange(i + 1, min(n, i + 1 + k))]
        )
        known = idxs[~np.isnan(values[idxs])]
        if len(known) < 2:
            continue
        try:
            poly = lagrange(known.astype(float), values[known])
            val = float(poly(float(i)))
            if np.isfinite(val):
                if clip is not None:
                    val = float(np.clip(val, clip[0], clip[1]))
                values[i] = val
        except Exception:
            continue
    med = np.nanmedian(values)
    if np.isnan(med):
        med = 0.0
    values[np.isnan(values)] = med
    return pd.Series(values, index=series.index)


def extract_title(name: str) -> str:
    title = pd.Series([name]).str.extract(r" ([A-Za-z]+)\.", expand=False).iloc[0]
    if pd.isna(title):
        return "Rare"
    mapping = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Royal",
        "Countess": "Royal",
        "Sir": "Royal",
        "Capt": "Rare",
        "Col": "Rare",
        "Don": "Rare",
        "Dr": "Rare",
        "Major": "Rare",
        "Rev": "Rare",
        "Jonkheer": "Rare",
        "Dona": "Rare",
    }
    title = mapping.get(title, title)
    if title not in {"Mr", "Miss", "Mrs", "Master", "Royal", "Rare"}:
        title = "Rare"
    return title


def add_raw_features(df: pd.DataFrame, ticket_freq: pd.Series | None = None) -> pd.DataFrame:
    out = df.copy()
    out["Title"] = out["Name"].map(extract_title)
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["FamilyBin"] = pd.cut(
        out["FamilySize"], bins=[0, 1, 4, 20], labels=["Alone", "Small", "Large"], include_lowest=True
    ).astype(str)
    out["HasCabin"] = out["Cabin"].notna().astype(int)
    deck = out["Cabin"].astype(str).str[0].replace({"n": "U"})
    out["CabinDeck"] = deck.where(deck.isin(list("ABCDEFGTU")), "U")
    out["TicketGroup"] = out["Ticket"].map(
        lambda x: "NUM" if str(x).replace(".", "").replace(" ", "").isdigit() else "PREF"
    )
    out["NameLen"] = out["Name"].str.len()
    out["SexPclass"] = out["Sex"].astype(str) + "_P" + out["Pclass"].astype(str)
    if ticket_freq is None:
        ticket_freq = out["Ticket"].value_counts()
    out["TicketFreq"] = out["Ticket"].map(ticket_freq).fillna(1).astype(float)
    return out
