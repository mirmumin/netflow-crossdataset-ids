"""Step 4: realistic setting - the target network has only BENIGN labels.

An operator can record normal traffic of their own network easily, but has
no labelled attacks. Attack knowledge can only come from other (source)
networks. Methods, all tested on the same target test flows:

  XGB-src        XGBoost trained on source networks (zero-shot)
  IForest-k      Isolation Forest fitted on the k target benign flows (anomaly detection, no source)
  XGB-pool-k     XGBoost on source data + the k target benign flows (up-weighted)
  Fusion-B-k     TabPFN whose context = k target benign flows (label 0) + source attack flows
                 (label 1); extra features = source XGBoost score and target anomaly score  (proposed)

Source = the other three networks (leave-one-network-out).

Usage:
    python scripts/run_benign_only.py              # 1 seed
    python scripts/run_benign_only.py --all-seeds
    python scripts/run_benign_only.py --learner rf # pipeline test without TabPFN
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402

from src.evaluate import binary_metrics  # noqa: E402
from src.features import LABEL_COL, feature_sets, to_matrix  # noqa: E402
from src.fusion import logit, make_fewshot_learner, predict_proba_chunked  # noqa: E402
from src.models import make_model  # noqa: E402
from src.splits import cap, split  # noqa: E402
from src.utils import load_config, resolve, set_seed  # noqa: E402

KEYS = ["seed", "target", "budget", "method"]


def anomaly_to_prob(iso: IsolationForest, X, ref) -> np.ndarray:
    """Map Isolation Forest scores to [0, 1] via the rank among the benign
    reference scores: 0.95 means more anomalous than 95% of known-benign flows."""
    s_ref = np.sort(-iso.score_samples(ref))
    return np.searchsorted(s_ref, -iso.score_samples(X), side="right") / len(s_ref)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--budgets", nargs="*", type=int, default=None)
    ap.add_argument("--learner", default=None)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--all-seeds", action="store_true")
    args = ap.parse_args()

    # ---- 1. Settings ----
    cfg = load_config()
    bc = cfg["benign_only"]
    budgets = args.budgets or bc["budgets"]
    learner = args.learner or bc["learner"]
    proc = resolve(args.processed_dir or cfg["paths"]["processed_dir"])
    res_dir = resolve(args.results_dir or cfg["paths"]["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)
    out_csv = res_dir / ("benign_only_raw.csv" if learner == "tabpfn" else f"benign_only_raw_{learner}.csv")

    keys = [k for k in cfg["datasets"] if (proc / f"{k}.parquet").exists()]
    data = {k: pd.read_parquet(proc / f"{k}.parquet") for k in keys}
    cols = set.intersection(*(set(d.columns) for d in data.values()))
    feats = feature_sets([c for c in next(iter(data.values())).columns if c in cols])["v3"]
    seeds = cfg["seeds"] if args.all_seeds else [cfg["seed"]]
    done = set()
    if out_csv.exists():
        done = set(map(tuple, pd.read_csv(out_csv)[KEYS].astype(str).values))

    for seed in seeds:
        set_seed(seed)
        rng = np.random.default_rng(seed)
        splits = {k: split(d, cfg["split"]["test_size"], seed) for k, d in data.items()}
        for tgt in args.targets or keys:
            src_list = [k for k in keys if k != tgt]
            tr_t, te_t = splits[tgt]
            te_t = cap(te_t, bc["max_test"], seed)
            X_te, y_te = to_matrix(te_t, feats), te_t[LABEL_COL].to_numpy()
            rows = []

            def record(method, k, p, secs):
                r = {"seed": seed, "target": tgt, "budget": k, "method": method,
                     "source": "+".join(src_list), "n_test": len(te_t), "seconds": round(secs, 2)}
                r.update(binary_metrics(y_te, p))
                rows.append(r)
                print(f"seed={seed} -> {tgt:<8} k={k:<5} {method:<11} F1={r['f1']:.3f} "
                      f"BalAcc={r['balanced_accuracy']:.3f} Rec={r['recall']:.3f} Spec={r['specificity']:.3f} ({secs:.1f}s)")

            todo = [k for k in budgets if any((str(seed), tgt, str(k), m) not in done
                                              for m in ["IForest-k", "XGB-pool-k", "Fusion-B-k"])]
            if not todo and (str(seed), tgt, "0", "XGB-src") in done:
                continue

            # ---- 2. Source model: XGBoost on the other three networks (has attack labels) ----
            per = bc["max_source_train"] // len(src_list)
            src_tr = pd.concat([cap(splits[s][0], per, seed) for s in src_list])
            X_src, y_src = to_matrix(src_tr, feats), src_tr[LABEL_COL].to_numpy()
            t0 = time.time()
            xgb_src = make_model("XGB", seed).fit(X_src, y_src)
            t_src = time.time() - t0
            p_src_te = xgb_src.predict_proba(X_te)[:, 1]
            if (str(seed), tgt, "0", "XGB-src") not in done:
                record("XGB-src", 0, p_src_te, t_src)

            # ---- 3. Target side: only k BENIGN flows are labelled ----
            benign_pool = tr_t[tr_t[LABEL_COL] == 0]
            for k in todo:
                ben = benign_pool.sample(n=min(k, len(benign_pool)), random_state=seed)
                X_b = to_matrix(ben, feats)

                # target-only anomaly detection
                t0 = time.time()
                iso = IsolationForest(n_estimators=300, random_state=seed).fit(X_b)
                record("IForest-k", k, anomaly_to_prob(iso, X_te, X_b), time.time() - t0)

                # pooled: source + target benign, target rows weighted to equal the source benign mass
                t0 = time.time()
                Xp = np.vstack([X_src, X_b])
                yp = np.concatenate([y_src, np.zeros(len(X_b), int)])
                w = np.ones(len(yp))
                w[len(y_src):] = max(1.0, (y_src == 0).sum() / len(X_b))
                pool = make_model("XGB", seed)
                pool.fit(Xp, yp, xgbclassifier__sample_weight=w)
                record("XGB-pool-k", k, pool.predict_proba(X_te)[:, 1], time.time() - t0)

                # proposed: TabPFN context = target benign + source attacks, with two knowledge features
                t0 = time.time()
                att_idx = np.flatnonzero(y_src == 1)
                n_att = min(bc["n_source_attacks"], len(att_idx), max(k, 50))
                X_a = X_src[rng.choice(att_idx, n_att, replace=False)]
                X_ctx = np.vstack([X_b, X_a])
                y_ctx = np.concatenate([np.zeros(len(X_b), int), np.ones(n_att, int)])

                def extra(X):
                    return np.column_stack([logit(xgb_src.predict_proba(X)[:, 1]),
                                            anomaly_to_prob(iso, X, X_b)])
                f = make_fewshot_learner(learner, seed, bc["tabpfn_version"])
                f.fit(np.hstack([X_ctx, extra(X_ctx)]), y_ctx)
                record("Fusion-B-k", k, predict_proba_chunked(f, np.hstack([X_te, extra(X_te)])),
                       time.time() - t0)

            pd.DataFrame(rows).to_csv(out_csv, mode="a", index=False, header=not out_csv.exists())

    df = pd.read_csv(out_csv)
    tab = df.groupby(["target", "budget", "method"])[["f1", "balanced_accuracy", "recall", "specificity"]].mean().round(4)
    tab.to_csv(res_dir / out_csv.name.replace("_raw", "_summary"))
    print("\n", tab.to_string())


if __name__ == "__main__":
    main()
