"""Step 6d: ROC curves and confusion matrices for the benign-only setting.

Reads the per-flow scores written by
    python scripts/run_benign_only.py --save-scores
(results/scores/benign_only_<seed>_<target>.npz) and draws
  fig_roc_k<k>      one ROC panel per target network, one curve per method
  fig_cm_k<k>       confusion matrices of Fusion-B and Fusion-B-v2 at the benign-q95 threshold

Usage:
    python scripts/plot_roc_cm.py [results_dir] [--budget 1000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve  # noqa: E402

from src.utils import load_config, resolve  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
TARGETS = [("unsw", "UNSW-NB15"), ("toniot", "ToN-IoT"), ("botiot", "BoT-IoT"),
           ("cic2018", "CSE-CIC-IDS2018")]
METHODS = [("XGB-src", "XGB-src (zero-shot)", "#7f7f7f", ":"),
           ("IForest-k", "IForest-k (target only)", "#a0a0a0", "--"),
           ("XGB-pool-k", "XGB-pool-k", "#6baed6", "-"),
           ("Fusion-B-k", "Fusion-B (proposed)", "#fd8d3c", "-"),
           ("Fusion-B-v2-k", "Fusion-B-v2 (proposed)", "#d62728", "-")]
CM_METHODS = ["Fusion-B-k", "Fusion-B-v2-k"]


def load_scores(res: Path, seed: int) -> dict:
    out = {}
    for key, _ in TARGETS:
        f = res / "scores" / f"benign_only_{seed}_{key}.npz"
        if f.exists():
            out[key] = dict(np.load(f))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=None)
    ap.add_argument("--budget", type=int, default=None, help="k; default: every k found")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    res = resolve(args.results_dir or cfg["paths"]["results_dir"])
    seed = args.seed if args.seed is not None else cfg["seed"]
    data = load_scores(res, seed)
    if not data:
        raise SystemExit(f"No score files in {res / 'scores'}; run run_benign_only.py --save-scores")
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    budgets = [args.budget] if args.budget else sorted(
        {int(k.split("|")[1]) for d in data.values() for k in d if k.count("|") == 1} - {0})
    targets = [t for t in TARGETS if t[0] in data]

    for k in budgets:
        # ---- ROC ----
        fig, axes = plt.subplots(1, len(targets), figsize=(2.9 * len(targets), 3.1), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (key, name) in zip(axes, targets):
            d = data[key]
            for m, label, color, ls in METHODS:
                s = d.get(f"{m}|{k}", d.get(f"{m}|0") if m == "XGB-src" else None)
                if s is None:
                    continue
                fpr, tpr, _ = roc_curve(d["y"], s)
                ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.8 if "Fusion" in m else 1.2,
                        label=f"{label} ({roc_auc_score(d['y'], s):.2f})")
            ax.plot([0, 1], [0, 1], color="black", linewidth=0.5, linestyle=":")
            ax.axvline(0.05, color="black", linewidth=0.5, alpha=0.4)
            ax.set_title(name)
            ax.set_xlabel("False-positive rate")
            ax.legend(fontsize=6, frameon=False, loc="lower right", title="AUROC", title_fontsize=6)
            ax.set_aspect("equal")
        axes[0].set_ylabel("True-positive rate (recall)")
        fig.suptitle(f"ROC curves, benign-only target, k = {k} (seed {seed}; "
                     "vertical line = 5% false alarms)")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"fig_roc_k{k}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out / f'fig_roc_k{k}.png'}")

        # ---- confusion matrices at the benign-q95 threshold ----
        fig, axes = plt.subplots(len(CM_METHODS), len(targets),
                                 figsize=(2.3 * len(targets), 2.3 * len(CM_METHODS)), squeeze=False)
        for r, m in enumerate(CM_METHODS):
            for c, (key, name) in enumerate(targets):
                ax = axes[r, c]
                d = data[key]
                if f"{m}|{k}" not in d:
                    ax.axis("off")
                    continue
                thr = float(d[f"{m}|{k}|threshold"])
                cm = confusion_matrix(d["y"], (d[f"{m}|{k}"] >= thr).astype(int), labels=[0, 1])
                share = cm / cm.sum(axis=1, keepdims=True)
                ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, f"{cm[i, j]:,}\n({share[i, j]:.1%})", ha="center", va="center",
                                fontsize=7, color="white" if share[i, j] > 0.6 else "black")
                ax.set_xticks([0, 1], ["benign", "attack"])
                ax.set_yticks([0, 1], ["benign", "attack"])
                if r == 0:
                    ax.set_title(name)
                if c == 0:
                    ax.set_ylabel(f"{m.replace('-k', '')}\ntrue class")
                if r == len(CM_METHODS) - 1:
                    ax.set_xlabel("predicted class")
        fig.suptitle(f"Confusion matrices at the benign-q95 threshold, k = {k} (seed {seed}); "
                     "row percentages")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"fig_cm_k{k}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out / f'fig_cm_k{k}.png'}")


if __name__ == "__main__":
    main()
