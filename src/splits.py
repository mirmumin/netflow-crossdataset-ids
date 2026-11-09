"""Train/test split per dataset and stratified row caps."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .features import ATTACK_COL


def _strata(df: pd.DataFrame) -> pd.Series:
    # classes with a single row cannot be stratified; merge them
    vc = df[ATTACK_COL].value_counts()
    return df[ATTACK_COL].where(df[ATTACK_COL].map(vc) >= 2, "_rare")


def split(df: pd.DataFrame, test_size: float, seed: int):
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=_strata(df))


def cap(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df
    keep, _ = train_test_split(df, train_size=n, random_state=seed, stratify=_strata(df))
    return keep
