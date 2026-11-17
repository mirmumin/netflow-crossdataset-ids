"""Step 2 (script version of notebooks/01_eda.ipynb): class balance and distribution shift.

  * fig_class_distribution  share of benign traffic and of each attack class per network
  * ks_shift_benign.csv     two-sample KS statistic per feature and network pair (benign flows only)
  * fig_ks_shift            heatmap of the most-shifted features + the mean over the six pairs

The KS statistic only depends on the order of the values, so it is computed on
the raw features (the log1p transform would give the same result).

Usage:
    python scripts/eda_shift.py [results_dir]
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

from src.features import ATTACK_COL, LABEL_COL, feature_sets  # noqa: E402
from src.utils import load_config, resolve  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
SHORT = {"unsw": "UNSW", "toniot": "ToN", "botiot": "BoT", "cic2018": "CIC2018"}
MAX_BENIGN = 100_000   # benign flows per network used for the KS test (speed)
TOP = 20               # features shown in the heatmap


def class_distribution(data: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(data), figsize=(3.1 * len(data), 3.4))
    axes = np.atleast_1d(axes)
    for ax, (k, df) in zip(axes, data.items()):
        vc = df[ATTACK_COL].value_counts(normalize=True).mul(100)
        vc = vc.sort_values()
        colors = ["#3aa66a" if str(c).lower() == "benign" else "#e05a5a" for c in vc.index]
        ax.barh(range(len(vc)), vc.values, color=colors, edgecolor="black", linewidth=0.3)
        ax.set_yticks(range(len(vc)), vc.index, fontsize=7)
        ax.set_xscale("log")
        ax.set_xlim(0.01, 150)
        for i, v in enumerate(vc.values):
            ax.text(v * 1.15, i, f"{v:.2f}" if v < 1 else f"{v:.1f}", va="center", fontsize=6.5)
        ax.set_title(f"{SHORT.get(k, k)} (n = {len(df):,})")
        ax.set_xlabel("Share of flows (%, log scale)")
    fig.suptitle("Class distribution of the stratified subsamples (green = benign, red = attack)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_class_distribution.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out / 'fig_class_distribution.png'}")


def ks_shift(data: dict, res: Path, out: Path, seed: int) -> pd.DataFrame:
    cols = set.intersection(*(set(d.columns) for d in data.values()))
    feats = [f for f in feature_sets([c for c in next(iter(data.values())).columns if c in cols])["v3"]]
    benign = {}
    for k, df in data.items():
        b = df[df[LABEL_COL] == 0]
        benign[k] = b.sample(n=min(MAX_BENIGN, len(b)), random_state=seed)
    rows = []
    for a, b in combinations(list(data), 2):
        for f in feats:
            rows.append({"pair": f"{SHORT.get(a, a)}–{SHORT.get(b, b)}", "feature": f,
                         "ks": ks_2samp(benign[a][f].to_numpy(float), benign[b][f].to_numpy(float)).statistic})
    ks = pd.DataFrame(rows).pivot(index="feature", columns="pair", values="ks")
    ks["mean"] = ks.mean(axis=1)
    ks = ks.sort_values("mean", ascending=False)
    ks.to_csv(res / "ks_shift_benign.csv")
    print(f"Wrote {res / 'ks_shift_benign.csv'}; mean KS over all features and pairs = "
          f"{ks.drop(columns='mean').values.mean():.3f}")
    print(ks.head(10).round(3).to_string())

    top = ks.head(TOP)
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * TOP + 1.4))
    im = ax.imshow(top.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    for i in range(top.shape[0]):
        for j in range(top.shape[1]):
            v = top.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="black" if v > 0.6 else "white")
    ax.set_yticks(range(len(top)), top.index, fontsize=7)
    ax.set_xticks(range(top.shape[1]), [c if c != "mean" else "MEAN" for c in top.columns],
                  rotation=35, ha="right")
    ax.axvline(top.shape[1] - 1.5, color="white", linewidth=2)
    fig.colorbar(im, ax=ax, shrink=0.8, label="KS statistic (benign flows)")
    ax.set_title(f"Distribution shift of the {TOP} most-shifted features between networks")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_ks_shift.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out / 'fig_ks_shift.png'}")
    return ks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=None)
    ap.add_argument("--processed-dir", default=None)
    args = ap.parse_args()
    cfg = load_config()
    proc = resolve(args.processed_dir or cfg["paths"]["processed_dir"])
    res = resolve(args.results_dir or cfg["paths"]["results_dir"])
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    data = {k: pd.read_parquet(proc / f"{k}.parquet") for k in cfg["datasets"]
            if (proc / f"{k}.parquet").exists()}
    if len(data) < 2:
        raise SystemExit(f"Need at least two processed datasets in {proc}")
    class_distribution(data, out)
    ks_shift(data, res, out, cfg["seed"])


if __name__ == "__main__":
    main()
