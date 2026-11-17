"""Step 6c: SHAP explainability of the source XGBoost models, compared with the KS shift.

Question: do the models rely on features that are strongly network-specific
(shortcuts)? For every network we train the Setting A XGBoost model (v3
features, configuration of Table 3) on its training split and compute exact
TreeSHAP values on a sample of its test flows, using XGBoost's built-in
TreeSHAP (pred_contribs=True; same algorithm as shap.TreeExplainer).

Outputs (results_dir):
  shap_importance.csv   mean |SHAP| per feature and network, normalised to sum to 1
  shap_vs_ks.csv        Spearman rank correlation and top-10 overlap with the KS shift
  figures/fig_shap_vs_ks.png/.pdf

Needs ks_shift_benign.csv from scripts/eda_shift.py for the comparison
(the importance table and figure are written without it).

Usage:
    python scripts/shap_analysis.py [results_dir] [--n-explain 2000]
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
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.features import LABEL_COL, feature_sets, to_matrix  # noqa: E402
from src.models import make_model  # noqa: E402
from src.splits import cap, split  # noqa: E402
from src.utils import load_config, resolve, set_seed  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9})
SHORT = {"unsw": "UNSW", "toniot": "ToN", "botiot": "BoT", "cic2018": "CIC2018"}
TOP = 15


def tree_shap(pipe, X: np.ndarray) -> np.ndarray:
    """Exact TreeSHAP values of the XGBoost step (inputs pass through the scaler first)."""
    from xgboost import DMatrix
    Z = pipe[:-1].transform(X)
    contrib = pipe[-1].get_booster().predict(DMatrix(Z), pred_contribs=True)
    return contrib[:, :-1]          # last column is the bias term


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=None)
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--n-explain", type=int, default=2000)
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    proc = resolve(args.processed_dir or cfg["paths"]["processed_dir"])
    res = resolve(args.results_dir or cfg["paths"]["results_dir"])
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)

    data = {k: pd.read_parquet(proc / f"{k}.parquet") for k in cfg["datasets"]
            if (proc / f"{k}.parquet").exists()}
    cols = set.intersection(*(set(d.columns) for d in data.values()))
    feats = feature_sets([c for c in next(iter(data.values())).columns if c in cols])["v3"]

    imp = {}
    for k, df in data.items():
        tr, te = split(df, cfg["split"]["test_size"], seed)
        tr = cap(tr, cfg["baselines"]["max_train"]["XGB"], seed)
        model = make_model("XGB", seed).fit(to_matrix(tr, feats), tr[LABEL_COL].to_numpy())
        ex = cap(te, args.n_explain, seed)
        sv = tree_shap(model, to_matrix(ex, feats))
        mean_abs = np.abs(sv).mean(axis=0)
        imp[SHORT.get(k, k)] = mean_abs / mean_abs.sum()
        top5 = pd.Series(imp[SHORT.get(k, k)], index=feats).nlargest(5)
        print(f"{k}: top features " + ", ".join(f"{f} ({v:.2f})" for f, v in top5.items()), flush=True)
    imp = pd.DataFrame(imp, index=feats)
    imp["mean"] = imp.mean(axis=1)
    imp = imp.sort_values("mean", ascending=False)
    imp.to_csv(res / "shap_importance.csv")
    print(f"Wrote {res / 'shap_importance.csv'}")

    ks_path = res / "ks_shift_benign.csv"
    ks = pd.read_csv(ks_path, index_col=0)["mean"] if ks_path.exists() else None
    if ks is not None:
        rows = []
        top_ks = set(ks.nlargest(10).index)
        for col in imp.columns:
            common = imp.index.intersection(ks.index)
            rho, p = spearmanr(imp.loc[common, col], ks.loc[common])
            top_imp = set(imp[col].nlargest(10).index)
            rows.append({"model": col, "spearman_rho": rho, "p": p,
                         "top10_overlap": len(top_imp & top_ks),
                         "shared_top10": "; ".join(sorted(top_imp & top_ks))})
        cmp_ = pd.DataFrame(rows)
        cmp_.to_csv(res / "shap_vs_ks.csv", index=False)
        print("SHAP importance vs KS shift:\n", cmp_.round(3).to_string(index=False))
    else:
        print(f"Skip SHAP-vs-KS comparison: {ks_path} not found (run scripts/eda_shift.py first)")

    # ---- figure: importance of the top features per network, next to their KS shift ----
    top = imp.head(TOP)
    nets = [c for c in top.columns if c != "mean"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.3 * TOP + 1.5), sharey=True,
                             gridspec_kw={"width_ratios": [len(nets), 1.4]})
    im = axes[0].imshow(top[nets].values, aspect="auto", cmap="Blues", vmin=0)
    for i in range(len(top)):
        for j, n in enumerate(nets):
            v = top[n].values[i]
            axes[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                         color="white" if v > top[nets].values.max() * 0.6 else "black")
    axes[0].set_xticks(range(len(nets)), [f"XGB trained\non {n}" for n in nets], fontsize=7.5)
    axes[0].set_yticks(range(len(top)), top.index, fontsize=7)
    fig.colorbar(im, ax=axes[0], shrink=0.7, label="share of mean |SHAP|")
    if ks is not None:
        vals = ks.reindex(top.index).values
        axes[1].barh(range(len(top)), vals, color="#e8833a", edgecolor="black", linewidth=0.3)
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("mean KS shift\n(benign, 6 pairs)")
    else:
        axes[1].axis("off")
    fig.suptitle("Most important features of the source XGBoost models (TreeSHAP) and their shift")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_shap_vs_ks.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out / 'fig_shap_vs_ks.png'}")


if __name__ == "__main__":
    main()
