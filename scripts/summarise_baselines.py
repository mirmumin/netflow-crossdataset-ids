"""Make the paper tables from results/baselines_raw.csv.

Writes:
  results/table_in_dataset.csv   mean +/- SD per model (source == target)
  results/table_cross_f1_<fs>.csv  4x4 F1 matrix per model (rows=source, cols=target)
  results/table_cross_summary.csv  mean cross-dataset metrics per model and feature set
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.utils import load_config, resolve  # noqa: E402

METRICS = ["accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy", "auroc"]


def mean_sd(g: pd.DataFrame) -> pd.Series:
    m, s = g[METRICS].mean(), g[METRICS].std().fillna(0)
    return pd.Series({k: f"{m[k]:.4f} ± {s[k]:.4f}" for k in METRICS})


def main(results_dir: str | None = None) -> None:
    cfg = load_config()
    res = resolve(results_dir or cfg["paths"]["results_dir"])
    df = pd.read_csv(res / "baselines_raw.csv")

    ind = df[df.source == df.target]
    t1 = ind.groupby(["feature_set", "source", "model"]).apply(mean_sd, include_groups=False)
    t1.to_csv(res / "table_in_dataset.csv")

    cross = df[df.source != df.target]
    for fs, g in cross.groupby("feature_set"):
        mat = g.groupby(["model", "source", "target"])["f1"].mean().unstack("target").round(4)
        mat.to_csv(res / f"table_cross_f1_{fs}.csv")
    t3 = cross.groupby(["feature_set", "model"])[METRICS].mean().round(4)
    t3.to_csv(res / "table_cross_summary.csv")

    print("In-dataset (mean ± SD):\n", t1.head(20).to_string())
    print("\nCross-dataset mean over 12 pairs:\n", t3.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
