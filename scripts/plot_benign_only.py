"""Paper figures for the benign-only setting (threshold rule benign-q95).

One figure per metric (F1, AUROC): a panel per target network, grouped bars
for k = 100 and k = 1000, one bar per method, error bar = SD over seeds.

Usage:
    python scripts/plot_benign_only.py [results_dir]
Output: results/figures/benign_only_f1.png/.pdf and benign_only_auroc.png/.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_config, resolve  # noqa: E402

METHODS = [("IForest-k", "IForest (target only)", "#9e9e9e"),
           ("XGB-pool-k", "XGB-pool", "#6baed6"),
           ("Fusion-B-k", "Fusion-B", "#fd8d3c"),
           ("Fusion-B-v2-k", "Fusion-B v2 (proposed)", "#d62728")]
TARGETS = [("unsw", "UNSW-NB15"), ("toniot", "ToN-IoT"), ("botiot", "BoT-IoT"), ("cic2018", "CSE-CIC-IDS2018")]


def main(results_dir: str | None = None) -> None:
    res = resolve(results_dir or load_config()["paths"]["results_dir"])
    df = pd.read_csv(res / "benign_only_raw.csv")
    df = df[(df["rule"] == "benign-q95") & (df["budget"] > 0)]
    stats = df.groupby(["target", "budget", "method"])[["f1", "auroc"]].agg(["mean", "std"])
    budgets = sorted(df["budget"].unique())
    fig_dir = res / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_seeds = df["seed"].nunique()

    for metric, label in [("f1", "F1"), ("auroc", "AUROC")]:
        targets = [t for t in TARGETS if t[0] in set(df["target"])]
        fig, axes = plt.subplots(1, len(targets), figsize=(3.2 * len(targets), 3.2), sharey=True)
        axes = np.atleast_1d(axes)
        width = 0.8 / len(METHODS)
        for ax, (key, name) in zip(axes, targets):
            for j, (m, mname, color) in enumerate(METHODS):
                means, sds = [], []
                for k in budgets:
                    try:
                        means.append(stats.loc[(key, k, m), (metric, "mean")])
                        sds.append(np.nan_to_num(stats.loc[(key, k, m), (metric, "std")]))
                    except KeyError:
                        means.append(np.nan)
                        sds.append(0)
                x = np.arange(len(budgets)) + (j - (len(METHODS) - 1) / 2) * width
                ax.bar(x, means, width, yerr=sds, capsize=2, color=color, label=mname,
                       edgecolor="black", linewidth=0.4)
            ax.set_xticks(np.arange(len(budgets)), [f"k={k}" for k in budgets])
            ax.set_title(name, fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel(label)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(METHODS), fontsize=8, frameon=False)
        fig.suptitle(f"Benign-only target: {label} (mean ± SD over {n_seeds} seeds)", fontsize=11)
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
        for ext in ["png", "pdf"]:
            fig.savefig(fig_dir / f"benign_only_{metric}.{ext}", dpi=300)
        plt.close(fig)
        print(f"Wrote {fig_dir / f'benign_only_{metric}.png'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
