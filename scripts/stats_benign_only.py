"""Step 7: is Fusion-B v2 better than the other methods, or is it chance?

Paired Wilcoxon signed-rank test on results/benign_only_raw.csv (threshold rule
benign-q95). Each pair = one (target network, k, seed) run, so 4 x 2 x 5 = 40
pairs with 5 seeds. For every other method we report:
  wins / ties / losses of Fusion-B-v2, median difference, Wilcoxon p-value and
  the Holm-corrected p-value (several comparisons are made at once).
Done for F1 and AUROC, over all runs and separately for k = 100 and k = 1000.

Usage:
    python scripts/stats_benign_only.py [results_dir]
Output: results/stats_benign_only.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_config, resolve  # noqa: E402

PROPOSED = "Fusion-B-v2-k"
RULE = "benign-q95"


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def compare(df: pd.DataFrame, metric: str, subset: str) -> list[dict]:
    wide = df.pivot_table(index=["target", "budget", "seed"], columns="method", values=metric)
    rows = []
    for other in [m for m in wide.columns if m != PROPOSED]:
        pair = wide[[PROPOSED, other]].dropna()
        d = pair[PROPOSED] - pair[other]
        nz = d[d.abs() > 1e-12]
        p = wilcoxon(nz).pvalue if len(nz) >= 2 else np.nan
        rows.append({"metric": metric, "subset": subset, "vs": other, "n_pairs": len(d),
                     "wins": int((d > 1e-12).sum()), "ties": int((d.abs() <= 1e-12).sum()),
                     "losses": int((d < -1e-12).sum()),
                     "mean_proposed": pair[PROPOSED].mean(), "mean_other": pair[other].mean(),
                     "median_diff": d.median(), "p": p})
    adj = holm([r["p"] if not np.isnan(r["p"]) else 1.0 for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = a
    return rows


def main(results_dir: str | None = None) -> None:
    res = resolve(results_dir or load_config()["paths"]["results_dir"])
    df = pd.read_csv(res / "benign_only_raw.csv")
    df = df[(df["rule"] == RULE) & (df["budget"] > 0)]
    rows = []
    for metric in ["f1", "auroc"]:
        rows += compare(df, metric, "all k")
        for k in sorted(df["budget"].unique()):
            rows += compare(df[df["budget"] == k], metric, f"k={k}")
    out = pd.DataFrame(rows).round(4)
    out.to_csv(res / "stats_benign_only.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{PROPOSED} vs others, paired Wilcoxon (rule {RULE}, seeds={df.seed.nunique()})\n")
    print(out.to_string(index=False))
    print(f"\nWrote {res / 'stats_benign_only.csv'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
