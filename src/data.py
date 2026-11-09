"""Loading and stratified subsampling of the UQ NetFlow CSVs.

The full v3 CSVs have 2-27 million rows. We read them in chunks and keep a
stratified subsample per attack class, so the whole pipeline runs on a free
Colab/Kaggle machine. Rare attack classes are kept whole (up to
min_per_class) so they do not vanish from the subsample.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import ATTACK_COL, LABEL_COL


def count_classes(csv_path: Path, chunksize: int) -> pd.Series:
    counts = pd.Series(dtype="int64")
    for chunk in pd.read_csv(csv_path, usecols=[ATTACK_COL], chunksize=chunksize):
        counts = counts.add(chunk[ATTACK_COL].value_counts(), fill_value=0)
    return counts.astype("int64").sort_values(ascending=False)


def class_fractions(counts: pd.Series, n_rows: int, min_per_class: int) -> dict[str, float]:
    """Sampling fraction per class.

    Classes with <= min_per_class rows are kept whole. The remaining budget is
    shared proportionally among the larger classes, but each still keeps at
    least min_per_class rows.
    """
    total = int(counts.sum())
    if total <= n_rows:
        return {k: 1.0 for k in counts.index}
    small = counts[counts <= min_per_class]
    large = counts[counts > min_per_class]
    budget = max(n_rows - int(small.sum()), 0)
    frac = budget / max(int(large.sum()), 1)
    out = {k: 1.0 for k in small.index}
    for k, c in large.items():
        keep = max(int(round(c * frac)), min_per_class)
        out[k] = min(1.0, keep / c)
    return out


def stratified_subsample(csv_path: Path, n_rows: int, min_per_class: int,
                         chunksize: int, seed: int) -> pd.DataFrame:
    counts = count_classes(csv_path, chunksize)
    fracs = class_fractions(counts, n_rows, min_per_class)
    rng = np.random.default_rng(seed)
    parts = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        u = rng.random(len(chunk))
        f = chunk[ATTACK_COL].map(fracs).fillna(1.0).to_numpy()
        parts.append(chunk[u < f])
    df = pd.concat(parts, ignore_index=True)
    df[LABEL_COL] = df[LABEL_COL].astype("int8")
    return df


def summarise(df: pd.DataFrame, name: str) -> pd.DataFrame:
    s = df[ATTACK_COL].value_counts().rename("n").to_frame()
    s["pct"] = (100 * s["n"] / len(df)).round(3)
    s.insert(0, "dataset", name)
    return s.reset_index().rename(columns={"index": ATTACK_COL})
