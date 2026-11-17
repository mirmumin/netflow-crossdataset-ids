"""Step 6b: the remaining paper figures, drawn from the result CSVs.

Every figure is skipped (with a message) if its input CSV does not exist yet.

  fig_cross_heatmap_<fs>   Setting A: 4x4 F1 matrix (rows = training network,
                           columns = test network) for each classical model
  fig_setting_b            Setting B: F1 of XGB-src, XGB-k, TabPFN-k and Fusion-k
  fig_budget_effect        Setting C: mean F1 / AUROC at k = 100 vs k = 1000
  fig_threshold_rule       Setting C: F1 and specificity with the 0.5 vs the benign-q95 threshold
  fig_wilcoxon             Setting C: wins / ties / losses of the proposed methods (40 paired runs)
  fig_runtime              Setting C: run time vs F1

Usage:
    python scripts/plot_paper_figures.py [results_dir]
Output: results/figures/<name>.png (300 dpi) and .pdf
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

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

TARGETS = [("unsw", "UNSW-NB15"), ("toniot", "ToN-IoT"), ("botiot", "BoT-IoT"),
           ("cic2018", "CSE-CIC-IDS2018")]
NET = dict(TARGETS)
# Setting C methods: code name -> (label in the paper, colour, marker)
C_METHODS = {
    "XGB-src":       ("XGB-src (zero-shot)", "#7f7f7f", "v"),
    "IForest-k":     ("IForest-k (target only)", "#b0b0b0", "s"),
    "XGB-src-BQN-k": ("XGB-src-BQN-k", "#9edae5", "D"),
    "XGB-pool-k":    ("XGB-pool-k", "#6baed6", "^"),
    "Fusion-B-k":    ("Fusion-B (proposed)", "#fd8d3c", "o"),
    "Fusion-B-v2-k": ("Fusion-B-v2 (proposed)", "#d62728", "o"),
}
Q_RULE = "benign-q95"


def save(fig, out: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out / name}.png")


def load(res: Path, name: str) -> pd.DataFrame | None:
    path = res / name
    if not path.exists():
        print(f"Skip: {path} not found")
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
def cross_heatmaps(res: Path, out: Path) -> None:
    """Setting A: one 4x4 F1 heatmap per model; the diagonal is the in-dataset result."""
    df = load(res, "baselines_raw.csv")
    if df is None:
        return
    order = [m for m in ["RF", "XGB", "LR", "SVM", "KNN", "ENS"] if m in set(df["model"])]
    names = {"XGB": "XGBoost", "ENS": "Ensemble"}
    keys = [k for k, _ in TARGETS if k in set(df["source"])]
    for fs, g in df.groupby("feature_set"):
        n_seeds = g["seed"].nunique()
        fig, axes = plt.subplots(2, 3, figsize=(10, 6.6), sharex=True, sharey=True)
        for ax, m in zip(axes.flat, order):
            mat = (g[g.model == m].groupby(["source", "target"])["f1"].mean()
                   .unstack("target").reindex(index=keys, columns=keys))
            im = ax.imshow(mat.values, vmin=0, vmax=1, cmap="RdYlGn")
            for i in range(len(keys)):
                for j in range(len(keys)):
                    v = mat.values[i, j]
                    ax.text(j, i, "–" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                            fontsize=8, fontweight="bold" if i == j else "normal",
                            color="white" if not np.isnan(v) and (v < 0.15 or v > 0.85) else "black")
            short = [NET[k].split("-")[0] if k != "cic2018" else "CIC2018" for k in keys]
            ax.set_xticks(range(len(keys)), short, rotation=30)
            ax.set_yticks(range(len(keys)), short)
            cross = mat.values[~np.eye(len(keys), dtype=bool)]
            ax.set_title(f"{names.get(m, m)} (cross mean {np.nanmean(cross):.2f})")
        for ax in axes.flat[len(order):]:
            ax.axis("off")
        fig.supxlabel("Test network (target)")
        fig.supylabel("Training network (source)")
        fig.colorbar(im, ax=axes, shrink=0.6, label="F1")
        fig.suptitle(f"In-dataset (diagonal) vs cross-dataset F1, {fs} features, "
                     f"mean over {n_seeds} seed(s)")
        save(fig, out, f"fig_cross_heatmap_{fs}")


# ---------------------------------------------------------------------------
def setting_b(res: Path, out: Path) -> None:
    """Setting B: target labels of both classes; does the source score help TabPFN?"""
    df = load(res, "fusion_raw.csv")
    if df is None:
        return
    if "mode" in df:
        df = df[df["mode"] == "loo"]
    methods = [("XGB-src", "XGB-src (zero-shot)", "#7f7f7f"), ("XGB-k", "XGB-k (target only)", "#6baed6"),
               ("TabPFN-k", "TabPFN-k (target only)", "#74c476"), ("Fusion-k", "Fusion-k", "#fd8d3c")]
    budgets = sorted(b for b in df["budget"].unique() if b > 0)
    targets = [t for t in TARGETS if t[0] in set(df["target"])]
    fig, axes = plt.subplots(1, len(targets), figsize=(2.6 * len(targets), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    w = 0.8 / len(methods)
    for ax, (key, name) in zip(axes, targets):
        g = df[df.target == key]
        for j, (m, label, color) in enumerate(methods):
            vals = []
            for k in budgets:
                s = g[(g.method == m) & ((g.budget == k) | ((m == "XGB-src") & (g.budget == 0)))]["f1"]
                vals.append((s.mean(), s.std() if len(s) > 1 else 0))
            x = np.arange(len(budgets)) + (j - (len(methods) - 1) / 2) * w
            ax.bar(x, [v[0] for v in vals], w, yerr=[np.nan_to_num(v[1]) for v in vals], capsize=2,
                   color=color, edgecolor="black", linewidth=0.4, label=label)
        ax.set_xticks(range(len(budgets)), [f"k={k}" for k in budgets])
        ax.set_title(name)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("F1")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(methods), frameon=False)
    fig.suptitle(f"Few labelled target flows (both classes), mean ± SD over {df['seed'].nunique()} seed(s)")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    save(fig, out, "fig_setting_b")


# ---------------------------------------------------------------------------
def benign_only(res: Path) -> pd.DataFrame | None:
    df = load(res, "benign_only_raw.csv")
    if df is None:
        return None
    return df[df["budget"] > 0]


def budget_effect(df: pd.DataFrame, out: Path) -> None:
    """Mean over the four targets, per seed; then mean ± SD over seeds."""
    q = df[df.rule == Q_RULE]
    per_seed = q.groupby(["method", "budget", "seed"])[["f1", "auroc"]].mean()
    agg = per_seed.groupby(["method", "budget"]).agg(["mean", "std"])
    budgets = sorted(q["budget"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3))
    for ax, (metric, label) in zip(axes, [("f1", "F1"), ("auroc", "AUROC")]):
        for m, (name, color, marker) in C_METHODS.items():
            if m not in agg.index.get_level_values(0):
                continue
            mu = [agg.loc[(m, k), (metric, "mean")] for k in budgets]
            sd = [np.nan_to_num(agg.loc[(m, k), (metric, "std")]) for k in budgets]
            ax.errorbar(range(len(budgets)), mu, yerr=sd, marker=marker, color=color, label=name,
                        capsize=3, linewidth=2 if "Fusion" in m else 1.2)
        ax.set_xticks(range(len(budgets)), [f"k = {k}" for k in budgets])
        ax.set_xlim(-0.3, len(budgets) - 0.7)
        ax.set_ylabel(f"Mean {label} over 4 targets")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Effect of the number of benign target flows (rule {Q_RULE})")
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    save(fig, out, "fig_budget_effect")


def threshold_rule(df: pd.DataFrame, out: Path) -> None:
    """Why a benign-calibrated threshold is needed: F1 and specificity under both rules."""
    methods = [m for m in C_METHODS if m != "XGB-src"]
    g = df[df.method.isin(methods)].groupby(["method", "rule"])[["f1", "specificity"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    x = np.arange(len(methods))
    for ax, (metric, label) in zip(axes, [("f1", "F1"), ("specificity", "Specificity (1 − false-alarm rate)")]):
        for j, (rule, color, hatch) in enumerate([("0.5", "#cccccc", "//"), (Q_RULE, "#d62728", "")]):
            vals = [g.loc[(m, rule), metric] if (m, rule) in g.index else np.nan for m in methods]
            ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=color, hatch=hatch, edgecolor="black",
                   linewidth=0.4, label=f"threshold {rule}")
        ax.set_xticks(x, [C_METHODS[m][0].replace(" (", "\n(").replace("XGB-src-BQN-k", "XGB-src-\nBQN-k")
                          for m in methods], fontsize=7)
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Fixed 0.5 threshold vs threshold from the benign target flows (mean over all runs)")
    fig.tight_layout()
    save(fig, out, "fig_threshold_rule")


def runtime(df: pd.DataFrame, out: Path) -> None:
    q = df[df.rule == Q_RULE]
    g = q.groupby(["method", "budget"])[["seconds", "f1"]].mean()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for (m, k), row in g.iterrows():
        if m not in C_METHODS:
            continue
        name, color, marker = C_METHODS[m]
        ax.scatter(row["seconds"], row["f1"], s=70 if k >= 1000 else 35, color=color, marker=marker,
                   edgecolor="black", linewidth=0.5, label=name if k == g.loc[m].index.max() else None)
    ax.set_xscale("log")
    ax.set_xlabel("Run time per target and seed (s, log scale)")
    ax.set_ylabel(f"Mean F1 ({Q_RULE})")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title("Cost vs benefit (small marker k = 100, large marker k = 1000)")
    save(fig, out, "fig_runtime")


def wilcoxon_chart(res: Path, out: Path) -> None:
    st = load(res, "stats_benign_only.csv")
    if st is None:
        return
    st = st[st["subset"] == "all k"]
    proposed = [m for m in ["Fusion-B-v2-k", "Fusion-B-k"] if m in set(st["method"])]
    fig, axes = plt.subplots(len(proposed), 2, figsize=(9, 2.2 * len(proposed)), squeeze=False)
    for r, pm in enumerate(proposed):
        for c, metric in enumerate(["f1", "auroc"]):
            ax = axes[r, c]
            s = st[(st.method == pm) & (st.metric == metric)].reset_index(drop=True)
            y = np.arange(len(s))
            ax.barh(y, s["wins"], color="#2ca02c", label="wins")
            ax.barh(y, s["ties"], left=s["wins"], color="#dddddd", label="ties")
            ax.barh(y, s["losses"], left=s["wins"] + s["ties"], color="#d62728", label="losses")
            n = int(s["n_pairs"].max())
            for i, row in s.iterrows():
                p = row["p_holm"]
                ptxt = "p<0.001" if p < 0.001 else f"p={p:.3f}"
                ax.text(n + 0.6, i, ptxt + (" *" if p < 0.05 else ""), va="center", fontsize=7.5)
            ax.set_yticks(y, ["vs " + C_METHODS.get(v, (v,))[0] for v in s["vs"]], fontsize=7.5)
            ax.set_xlim(0, n * 1.3)
            ax.axvline(n / 2, color="black", linewidth=0.6, linestyle=":")
            ax.set_title(f"{C_METHODS[pm][0]}: {'F1' if metric == 'f1' else 'AUROC'} ({n} paired runs)")
            ax.invert_yaxis()
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower right", ncol=3, bbox_to_anchor=(1, 1.12))
    fig.suptitle("Paired Wilcoxon signed-rank tests, Holm-corrected (* = p < 0.05)", y=1.02)
    fig.tight_layout()
    save(fig, out, "fig_wilcoxon")


def main(results_dir: str | None = None) -> None:
    res = resolve(results_dir or load_config()["paths"]["results_dir"])
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    cross_heatmaps(res, out)
    setting_b(res, out)
    df = benign_only(res)
    if df is not None:
        budget_effect(df, out)
        threshold_rule(df, out)
        runtime(df, out)
    wilcoxon_chart(res, out)


if __name__ == "__main__":
    main(*sys.argv[1:])
