"""Step 3: proposed Fusion model vs few-shot and zero-shot baselines.

For every TARGET network, label budget k and seed, compare on the same
target test flows:
  XGB-src      XGBoost trained on the source network(s), no target labels (zero-shot)
  XGB-k        XGBoost trained only on the k labelled target flows
  TabPFN-k     TabPFN on the k labelled target flows
  Fusion-k     TabPFN on the k target flows + source XGBoost score as a feature (proposed)

Mode "loo" (default): source = the other three networks (leave-one-network-out).
Mode "pair": every source -> target pair separately (12 pairs).

Usage (Colab, GPU runtime recommended):
    python scripts/run_fusion.py                    # 1 seed, quick demo
    python scripts/run_fusion.py --all-seeds        # final numbers
    python scripts/run_fusion.py --learner rf       # no TabPFN, pipeline test only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluate import binary_metrics  # noqa: E402
from src.features import LABEL_COL, feature_sets, to_matrix  # noqa: E402
from src.fusion import logit, make_fewshot_learner, predict_proba_chunked  # noqa: E402
from src.models import make_model  # noqa: E402
from src.splits import cap, split  # noqa: E402
from src.utils import load_config, resolve, set_seed  # noqa: E402

KEYS = ["seed", "mode", "source", "target", "budget", "method"]


def sample_labelled(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """k target flows with both classes present (at least 10% of the minority, max 50%)."""
    rng = np.random.default_rng(seed)
    pos, neg = df[df[LABEL_COL] == 1], df[df[LABEL_COL] == 0]
    rate = len(pos) / len(df)
    n_pos = int(round(k * min(max(rate, 0.1), 0.9)))
    n_pos = min(n_pos, len(pos))
    n_neg = min(k - n_pos, len(neg))
    idx = np.concatenate([rng.choice(pos.index, n_pos, replace=False),
                          rng.choice(neg.index, n_neg, replace=False)])
    return df.loc[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--mode", choices=["loo", "pair"], default=None)
    ap.add_argument("--budgets", nargs="*", type=int, default=None)
    ap.add_argument("--learner", default=None)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--all-seeds", action="store_true")
    args = ap.parse_args()

    # ---- 1. Settings ----
    cfg = load_config()
    fc = cfg["fusion"]
    mode = args.mode or fc["mode"]
    budgets = args.budgets or fc["budgets"]
    learner = args.learner or fc["learner"]
    proc = resolve(args.processed_dir or cfg["paths"]["processed_dir"])
    res_dir = resolve(args.results_dir or cfg["paths"]["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)
    out_csv = res_dir / ("fusion_raw.csv" if learner == "tabpfn" else f"fusion_raw_{learner}.csv")

    keys = [k for k in cfg["datasets"] if (proc / f"{k}.parquet").exists()]
    data = {k: pd.read_parquet(proc / f"{k}.parquet") for k in keys}
    cols = set.intersection(*(set(d.columns) for d in data.values()))
    feats = feature_sets([c for c in next(iter(data.values())).columns if c in cols])["v3"]
    seeds = cfg["seeds"] if args.all_seeds else [cfg["seed"]]
    targets = args.targets or keys

    done = set()
    if out_csv.exists():
        done = set(map(tuple, pd.read_csv(out_csv)[KEYS].astype(str).values))

    # ---- 2. Main loop: for each TARGET network ----
    for seed in seeds:
        set_seed(seed)
        splits = {k: split(d, cfg["split"]["test_size"], seed) for k, d in data.items()}
        for tgt in targets:
            sources = [[k for k in keys if k != tgt]] if mode == "loo" else [[k] for k in keys if k != tgt]
            tr_t, te_t = splits[tgt]
            te_t = cap(te_t, fc["max_test"], seed)
            X_te = to_matrix(te_t, feats)
            y_te = te_t[LABEL_COL].to_numpy()
            for src_list in sources:
                src = "+".join(src_list)
                rows = []

                def record(method, k, p, secs):
                    r = {"seed": seed, "mode": mode, "source": src, "target": tgt,
                         "budget": k, "method": method, "n_test": len(te_t),
                         "seconds": round(secs, 2)}
                    r.update(binary_metrics(y_te, p))
                    rows.append(r)
                    print(f"seed={seed} {src:>22} -> {tgt:<8} k={k:<5} {method:<9} "
                          f"F1={r['f1']:.3f} BalAcc={r['balanced_accuracy']:.3f} ({secs:.1f}s)")

                todo_budgets = [k for k in budgets
                                if any((str(seed), mode, src, tgt, str(k), m) not in done
                                       for m in ["XGB-k", "TabPFN-k", "Fusion-k"])]
                need_src = todo_budgets or (str(seed), mode, src, tgt, "0", "XGB-src") not in done
                if not need_src:
                    continue

                # ---- 2a. Train XGBoost on the SOURCE network(s) (equal share of rows each) ----
                t0 = time.time()
                per = fc["max_source_train"] // len(src_list)
                src_tr = pd.concat([cap(splits[s][0], per, seed) for s in src_list])
                xgb_src = make_model("XGB", seed)
                xgb_src.fit(to_matrix(src_tr, feats), src_tr[LABEL_COL].to_numpy())
                t_src = time.time() - t0
                p_src_te = xgb_src.predict_proba(X_te)[:, 1]
                if (str(seed), mode, src, tgt, "0", "XGB-src") not in done:
                    record("XGB-src", 0, p_src_te, t_src)

                # ---- 2b. For each label budget k: pretend only k target flows are labelled ----
                for k in todo_budgets:
                    lab = sample_labelled(tr_t, k, seed)
                    X_l, y_l = to_matrix(lab, feats), lab[LABEL_COL].to_numpy()

                    # baseline: XGBoost trained on the k target labels only
                    t0 = time.time()
                    xgb_k = make_model("XGB", seed).fit(X_l, y_l)
                    record("XGB-k", k, xgb_k.predict_proba(X_te)[:, 1], time.time() - t0)

                    # baseline: TabPFN on the k target labels only
                    # (features are already log1p-transformed by to_matrix; TabPFN needs no scaling)
                    Z_l, Z_te = X_l, X_te
                    t0 = time.time()
                    m = make_fewshot_learner(learner, seed, fc["tabpfn_version"]).fit(Z_l, y_l)
                    record("TabPFN-k" if learner == "tabpfn" else f"{learner}-k", k,
                           predict_proba_chunked(m, Z_te), time.time() - t0)

                    # proposed: + source model's attack score as an extra feature
                    s_l = logit(xgb_src.predict_proba(X_l)[:, 1])[:, None]
                    s_te = logit(p_src_te)[:, None]
                    t0 = time.time()
                    f = make_fewshot_learner(learner, seed, fc["tabpfn_version"])
                    f.fit(np.hstack([Z_l, s_l]), y_l)
                    record("Fusion-k", k, predict_proba_chunked(f, np.hstack([Z_te, s_te])),
                           time.time() - t0)

                pd.DataFrame(rows).to_csv(out_csv, mode="a", index=False, header=not out_csv.exists())
                for r in rows:
                    done.add(tuple(map(str, (r[c] for c in KEYS))))

    df = pd.read_csv(out_csv)
    tab = (df.groupby(["mode", "target", "budget", "method"])[["f1", "balanced_accuracy", "recall", "specificity"]]
           .mean().round(4))
    tab.to_csv(res_dir / out_csv.name.replace("_raw", "_summary"))
    print("\n", tab.to_string())


if __name__ == "__main__":
    main()
