"""Step 5: realistic setting - the target network has only BENIGN labels.

An operator can record normal traffic of their own network easily, but has
no labelled attacks. Attack knowledge can only come from other (source)
networks. Methods, all tested on the same target test flows:

  XGB-src         XGBoost trained on source networks (zero-shot)
  IForest-k       Isolation Forest fitted on the k target benign flows (anomaly detection, no source).
                  A flow is called an attack if it is more anomalous than 95% of known benign flows.
  XGB-pool-k      XGBoost on source data + the k target benign flows (up-weighted)
  Fusion-B-k      TabPFN whose context = k target benign flows (label 0) + source attack flows
                  (label 1); extra features = source XGBoost score and target anomaly score
  XGB-src-BQN-k   XGBoost zero-shot, but every network's features are first aligned to its own
                  benign traffic (benign-anchored quantile normalisation, src/align.py)
  Fusion-B-v2-k   Fusion-B on the aligned features  (proposed)

Leakage rules (v2 of this script):
  * The source attack flows put in TabPFN's context are held out from the source
    XGBoost's training data, so their XGBoost score is out-of-sample, like the
    score of a target test flow.
  * The anomaly score of the k benign context flows is cross-fitted (each flow is
    scored by an Isolation Forest that did not see it).

Source = the other three networks (leave-one-network-out).
Results: results/benign_only_v2_raw.csv (one row per seed/target/k/method).

Usage:
    python scripts/run_benign_only.py              # 1 seed
    python scripts/run_benign_only.py --all-seeds  # 5 seeds, for the paper
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
from sklearn.model_selection import KFold  # noqa: E402

from src.align import BenignQuantileAligner  # noqa: E402
from src.evaluate import binary_metrics  # noqa: E402
from src.features import LABEL_COL, feature_sets, to_matrix  # noqa: E402
from src.fusion import logit, make_fewshot_learner, predict_proba_chunked  # noqa: E402
from src.models import make_model  # noqa: E402
from src.splits import cap, split  # noqa: E402
from src.utils import load_config, resolve, set_seed  # noqa: E402

KEYS = ["seed", "target", "budget", "method"]
K_METHODS = ["IForest-k", "XGB-pool-k", "Fusion-B-k", "XGB-src-BQN-k", "Fusion-B-v2-k"]
IFOREST_THRESHOLD = 0.95  # attack = more anomalous than 95% of known benign flows


class BenignAnomaly:
    """Isolation Forest on the k target benign flows.

    score(X) = fraction of known-benign reference scores that X is more anomalous
    than (0.95 -> more anomalous than 95% of benign). The reference scores are
    cross-fitted, so they are not optimistically low.
    """

    def __init__(self, X_b, seed: int, n_folds: int = 5):
        mk = lambda: IsolationForest(n_estimators=300, random_state=seed)  # noqa: E731
        self.iso = mk().fit(X_b)
        self.oof = np.empty(len(X_b))
        for tr, te in KFold(min(n_folds, len(X_b)), shuffle=True, random_state=seed).split(X_b):
            self.oof[te] = -mk().fit(X_b[tr]).score_samples(X_b[te])
        self.ref = np.sort(self.oof)

    def _rank(self, raw):
        return np.searchsorted(self.ref, raw, side="right") / len(self.ref)

    def score(self, X):
        return self._rank(-self.iso.score_samples(X))

    def score_train(self):
        """Out-of-fold score of the k benign flows themselves (for TabPFN's context)."""
        return self._rank(self.oof)


def align_per_network(X, nets, benign_mask, seed):
    """Align each network's rows with a quantile map fitted on that network's benign rows."""
    out = np.empty_like(X, dtype="float32")
    for n in np.unique(nets):
        m = nets == n
        out[m] = BenignQuantileAligner(seed=seed).fit(X[m & benign_mask]).transform(X[m])
    return out


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
    out_csv = res_dir / ("benign_only_v2_raw.csv" if learner == "tabpfn"
                         else f"benign_only_v2_raw_{learner}.csv")

    keys = [k for k in cfg["datasets"] if (proc / f"{k}.parquet").exists()]
    data = {k: pd.read_parquet(proc / f"{k}.parquet") for k in keys}
    cols = set.intersection(*(set(d.columns) for d in data.values()))
    feats = feature_sets([c for c in next(iter(data.values())).columns if c in cols])["v3"]
    seeds = cfg["seeds"] if args.all_seeds else [cfg["seed"]]
    done = set()
    if out_csv.exists():  # resume: skip rows already computed
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

            def record(method, k, p, secs, threshold=0.5):
                r = {"seed": seed, "target": tgt, "budget": k, "method": method,
                     "source": "+".join(src_list), "n_test": len(te_t), "threshold": threshold,
                     "seconds": round(secs, 2)}
                r.update(binary_metrics(y_te, p, threshold))
                rows.append(r)
                print(f"seed={seed} -> {tgt:<8} k={k:<5} {method:<14} F1={r['f1']:.3f} "
                      f"BalAcc={r['balanced_accuracy']:.3f} Rec={r['recall']:.3f} "
                      f"Spec={r['specificity']:.3f} AUROC={r['auroc']:.3f} ({secs:.1f}s)", flush=True)

            todo = [k for k in budgets if any((str(seed), tgt, str(k), m) not in done for m in K_METHODS)]
            if not todo and (str(seed), tgt, "0", "XGB-src") in done:
                continue

            # ---- 2. Source data (other three networks, attack labels known) ----
            per = bc["max_source_train"] // len(src_list)
            parts = [cap(splits[s][0], per, seed) for s in src_list]
            src_tr = pd.concat(parts)
            nets = np.concatenate([[s] * len(p) for s, p in zip(src_list, parts)])
            X_src, y_src = to_matrix(src_tr, feats), src_tr[LABEL_COL].to_numpy()

            # Hold out some source attack flows for TabPFN's context; XGBoost never trains on them.
            att_idx = np.flatnonzero(y_src == 1)
            n_res = min(bc["n_source_attacks"], len(att_idx) // 2)
            reserve = rng.choice(att_idx, n_res, replace=False)
            fit_mask = np.ones(len(y_src), bool)
            fit_mask[reserve] = False

            # ---- 3. Source models: raw features and aligned features ----
            t0 = time.time()
            xgb_src = make_model("XGB", seed).fit(X_src[fit_mask], y_src[fit_mask])
            t_src = time.time() - t0
            if (str(seed), tgt, "0", "XGB-src") not in done:
                record("XGB-src", 0, xgb_src.predict_proba(X_te)[:, 1], t_src)

            t0 = time.time()
            X_src_al = align_per_network(X_src, nets, y_src == 0, seed)
            xgb_al = make_model("XGB", seed).fit(X_src_al[fit_mask], y_src[fit_mask])
            t_al = time.time() - t0

            # ---- 4. Target side: only k BENIGN flows are labelled ----
            benign_pool = tr_t[tr_t[LABEL_COL] == 0]
            for k in todo:
                ben = benign_pool.sample(n=min(k, len(benign_pool)), random_state=seed)
                X_b = to_matrix(ben, feats)
                n_att = min(n_res, max(k, 50))
                ctx_att = reserve[:n_att]             # reserve is already in random order
                y_ctx = np.concatenate([np.zeros(len(X_b), int), np.ones(n_att, int)])

                # 4a. target-only anomaly detection
                t0 = time.time()
                anom = BenignAnomaly(X_b, seed)
                s_te = anom.score(X_te)
                record("IForest-k", k, s_te, time.time() - t0, threshold=IFOREST_THRESHOLD)
                anom_ctx = np.concatenate([anom.score_train(), anom.score(X_src[ctx_att])])

                # 4b. pooled: source + target benign, target rows weighted to equal source benign mass
                t0 = time.time()
                Xp = np.vstack([X_src[fit_mask], X_b])
                yp = np.concatenate([y_src[fit_mask], np.zeros(len(X_b), int)])
                w = np.ones(len(yp))
                w[fit_mask.sum():] = max(1.0, (y_src[fit_mask] == 0).sum() / len(X_b))
                pool = make_model("XGB", seed)
                pool.fit(Xp, yp, xgbclassifier__sample_weight=w)
                record("XGB-pool-k", k, pool.predict_proba(X_te)[:, 1], time.time() - t0)

                # 4c. Fusion-B: TabPFN context = target benign + held-out source attacks
                t0 = time.time()
                X_ctx = np.vstack([X_b, X_src[ctx_att]])
                f = make_fewshot_learner(learner, seed, bc["tabpfn_version"])
                f.fit(np.column_stack([X_ctx, logit(xgb_src.predict_proba(X_ctx)[:, 1]), anom_ctx]), y_ctx)
                Z_te = np.column_stack([X_te, logit(xgb_src.predict_proba(X_te)[:, 1]), s_te])
                record("Fusion-B-k", k, predict_proba_chunked(f, Z_te), time.time() - t0)

                # 4d. aligned zero-shot: target features mapped with the k target benign flows
                t0 = time.time()
                al_t = BenignQuantileAligner(seed=seed).fit(X_b)
                X_te_al, X_b_al = al_t.transform(X_te), al_t.transform(X_b)
                p_al_te = xgb_al.predict_proba(X_te_al)[:, 1]
                record("XGB-src-BQN-k", k, p_al_te, t_al + time.time() - t0)

                # 4e. Fusion-B v2 (proposed): Fusion-B on aligned features
                t0 = time.time()
                X_ctx_al = np.vstack([X_b_al, X_src_al[ctx_att]])
                f = make_fewshot_learner(learner, seed, bc["tabpfn_version"])
                f.fit(np.column_stack([X_ctx_al, logit(xgb_al.predict_proba(X_ctx_al)[:, 1]), anom_ctx]), y_ctx)
                Z_te = np.column_stack([X_te_al, logit(p_al_te), s_te])
                record("Fusion-B-v2-k", k, predict_proba_chunked(f, Z_te), t_al + time.time() - t0)

            pd.DataFrame(rows).to_csv(out_csv, mode="a", index=False, header=not out_csv.exists())

    # ---- 5. Summary table (mean over seeds) ----
    df = pd.read_csv(out_csv)
    tab = (df.groupby(["target", "budget", "method"])
           [["f1", "balanced_accuracy", "recall", "specificity", "auroc"]].agg(["mean", "std"]).round(4))
    tab.to_csv(res_dir / out_csv.name.replace("_raw", "_summary"))
    print("\n", df.groupby(["target", "budget", "method"])
          [["f1", "balanced_accuracy", "recall", "specificity", "auroc"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
