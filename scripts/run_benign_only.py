"""Step 5: realistic setting - the target network has only BENIGN labels.

An operator can record normal traffic of their own network easily, but has
no labelled attacks. Attack knowledge can only come from other (source)
networks. Methods, all tested on the same target test flows:

  XGB-src         XGBoost trained on source networks (zero-shot)
  IForest-k       Isolation Forest fitted on the k target benign flows (anomaly detection, no source).
  XGB-pool-k      XGBoost on source data + the k target benign flows (up-weighted)
  Fusion-B-k      (proposed) TabPFN whose context = k target benign flows (label 0) + source attack flows
                  (label 1); extra features = source XGBoost score and target anomaly score
  XGB-src-BQN-k   XGBoost zero-shot, but every network's features are first aligned to its own
                  benign traffic (benign-anchored quantile normalisation, src/align.py)
  Fusion-B-v2-k   Fusion-B on the aligned features (proposed variant)

Decision threshold. Every method gives a score; a flow is called an attack if
its score is >= a threshold. Two rules are reported for every method (column "rule"):
  0.5         the usual fixed cut-off (meaningless for anomaly scores)
  benign-q95  the 95th percentile of the method's scores on the k known benign
              target flows, each scored by a model that did not see it
              (cross-fitted). Needs benign labels only, so it is allowed in this
              setting, and targets a ~5% false-alarm rate.
AUROC does not depend on the threshold.

Leakage rules:
  * The source attack flows put in TabPFN's context are held out from the source
    XGBoost's training data, so their XGBoost score is out-of-sample, like the
    score of a target test flow.
  * The anomaly score of the k benign context flows is cross-fitted (each flow is
    scored by an Isolation Forest that did not see it).

Source = the other three networks (leave-one-network-out).
Results: results/benign_only_raw.csv (one row per seed/target/k/method/rule).

With --save-scores the run also writes every method's score for every test
flow (results/scores/benign_only_<seed>_<target>.npz, used for ROC curves and
confusion matrices) and appends its metrics to benign_only_raw_scorerun.csv
instead, so the main results file is not duplicated.

Usage:
    python scripts/run_benign_only.py              # 1 seed
    python scripts/run_benign_only.py --save-scores  # 1 seed, keep per-flow scores
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

KEYS = ["seed", "target", "budget", "method", "rule"]
K_METHODS = ["XGB-src", "IForest-k", "XGB-pool-k", "Fusion-B-k", "XGB-src-BQN-k", "Fusion-B-v2-k"]
BENIGN_Q = 0.95   # threshold = 95th percentile of cross-fitted scores on known benign flows
N_FOLDS = 3       # folds for the cross-fitted benign scores


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
    ap.add_argument("--save-scores", action="store_true")
    args = ap.parse_args()

    # ---- 1. Settings ----
    cfg = load_config()
    bc = cfg["benign_only"]
    budgets = args.budgets or bc["budgets"]
    learner = args.learner or bc["learner"]
    proc = resolve(args.processed_dir or cfg["paths"]["processed_dir"])
    res_dir = resolve(args.results_dir or cfg["paths"]["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)
    out_csv = res_dir / ("benign_only_raw.csv" if learner == "tabpfn"
                         else f"benign_only_raw_{learner}.csv")
    if args.save_scores:
        out_csv = out_csv.with_name(out_csv.stem + "_scorerun.csv")
        (res_dir / "scores").mkdir(exist_ok=True)

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
            scores = {"y": y_te}   # per-flow scores, saved with --save-scores

            def record(method, k, p, secs, benign_oof=None):
                """Score the test flows at 0.5 and (if benign scores given) at the benign-q95 threshold."""
                scores[f"{method}|{k}"] = np.asarray(p, dtype="float32")
                rules = [("0.5", 0.5)]
                if benign_oof is not None:
                    rules.append((f"benign-q{int(BENIGN_Q * 100)}", float(np.quantile(benign_oof, BENIGN_Q))))
                    scores[f"{method}|{k}|threshold"] = np.float32(rules[-1][1])
                for rule, thr in rules:
                    r = {"seed": seed, "target": tgt, "budget": k, "method": method, "rule": rule,
                         "threshold": thr, "source": "+".join(src_list), "n_test": len(te_t),
                         "seconds": round(secs, 2)}
                    r.update(binary_metrics(y_te, p, thr))
                    rows.append(r)
                    print(f"seed={seed} -> {tgt:<8} k={k:<5} {method:<14} {rule:<10} F1={r['f1']:.3f} "
                          f"BalAcc={r['balanced_accuracy']:.3f} Rec={r['recall']:.3f} "
                          f"Spec={r['specificity']:.3f} AUROC={r['auroc']:.3f} ({secs:.1f}s)", flush=True)

            todo = [k for k in budgets if any((str(seed), tgt, str(k), m, "benign-q95") not in done
                                              for m in K_METHODS)]
            if not todo and (str(seed), tgt, "0", "XGB-src", "0.5") in done:
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
            p_src_te = xgb_src.predict_proba(X_te)[:, 1]
            if (str(seed), tgt, "0", "XGB-src", "0.5") not in done:
                record("XGB-src", 0, p_src_te, t_src)          # zero-shot, no target labels at all

            t0 = time.time()
            X_src_al = align_per_network(X_src, nets, y_src == 0, seed)
            xgb_al = make_model("XGB", seed).fit(X_src_al[fit_mask], y_src[fit_mask])
            t_al = time.time() - t0

            # ---- 4. Target side: only k BENIGN flows are labelled ----
            benign_pool = tr_t[tr_t[LABEL_COL] == 0]
            for k in todo:
                ben = benign_pool.sample(n=min(k, len(benign_pool)), random_state=seed)
                X_b = to_matrix(ben, feats)
                folds = list(KFold(N_FOLDS, shuffle=True, random_state=seed).split(X_b))
                n_att = min(n_res, max(k, 50))
                X_att, X_att_al = X_src[reserve[:n_att]], X_src_al[reserve[:n_att]]  # reserve is shuffled

                def oof(score_fn):
                    """Cross-fitted scores of the k benign flows: score_fn(train_idx, eval_idx)."""
                    out = np.empty(len(X_b))
                    for tr, te in folds:
                        out[te] = score_fn(tr, te)
                    return out

                # 4a. source model with a target-calibrated threshold (k benign used for the threshold only)
                record("XGB-src", k, p_src_te, t_src, benign_oof=xgb_src.predict_proba(X_b)[:, 1])

                # 4b. target-only anomaly detection
                t0 = time.time()
                anom = BenignAnomaly(X_b, seed)
                s_te, s_b, s_att = anom.score(X_te), anom.score_train(), anom.score(X_att)
                record("IForest-k", k, s_te, time.time() - t0, benign_oof=s_b)

                # 4c. pooled: source + target benign, target rows weighted to equal source benign mass
                t0 = time.time()

                def fit_pool(idx):
                    Xp = np.vstack([X_src[fit_mask], X_b[idx]])
                    yp = np.concatenate([y_src[fit_mask], np.zeros(len(idx), int)])
                    w = np.ones(len(yp))
                    w[fit_mask.sum():] = max(1.0, (y_src[fit_mask] == 0).sum() / len(idx))
                    m = make_model("XGB", seed)
                    return m.fit(Xp, yp, xgbclassifier__sample_weight=w)
                p_pool = fit_pool(np.arange(len(X_b))).predict_proba(X_te)[:, 1]
                b_pool = oof(lambda tr, te: fit_pool(tr).predict_proba(X_b[te])[:, 1])
                record("XGB-pool-k", k, p_pool, time.time() - t0, benign_oof=b_pool)

                # 4d. Fusion-B: TabPFN context = target benign + held-out source attacks (raw features)
                def fusion_b(Xb, ab, Xe, ae):
                    ctx = np.vstack([Xb, X_att])
                    y = np.concatenate([np.zeros(len(Xb), int), np.ones(len(X_att), int)])
                    f = make_fewshot_learner(learner, seed, bc["tabpfn_version"])
                    f.fit(np.column_stack([ctx, logit(xgb_src.predict_proba(ctx)[:, 1]),
                                           np.concatenate([ab, s_att])]), y)
                    return predict_proba_chunked(f, np.column_stack(
                        [Xe, logit(xgb_src.predict_proba(Xe)[:, 1]), ae]))
                t0 = time.time()
                p_fb = fusion_b(X_b, s_b, X_te, s_te)
                b_fb = oof(lambda tr, te: fusion_b(X_b[tr], s_b[tr], X_b[te], s_b[te]))
                record("Fusion-B-k", k, p_fb, time.time() - t0, benign_oof=b_fb)

                # 4e. aligned zero-shot: target features mapped with the target benign flows
                def bqn(Xb, Xe):
                    return xgb_al.predict_proba(BenignQuantileAligner(seed=seed).fit(Xb).transform(Xe))[:, 1]
                t0 = time.time()
                p_bqn = bqn(X_b, X_te)
                b_bqn = oof(lambda tr, te: bqn(X_b[tr], X_b[te]))
                record("XGB-src-BQN-k", k, p_bqn, t_al + time.time() - t0, benign_oof=b_bqn)

                # 4f. Fusion-B v2 (proposed variant): Fusion-B on benign-aligned features
                def fusion_v2(Xb, ab, Xe, ae):
                    al = BenignQuantileAligner(seed=seed).fit(Xb)
                    ctx = np.vstack([al.transform(Xb), X_att_al])
                    y = np.concatenate([np.zeros(len(Xb), int), np.ones(len(X_att_al), int)])
                    f = make_fewshot_learner(learner, seed, bc["tabpfn_version"])
                    f.fit(np.column_stack([ctx, logit(xgb_al.predict_proba(ctx)[:, 1]),
                                           np.concatenate([ab, s_att])]), y)
                    Ze = al.transform(Xe)
                    return predict_proba_chunked(f, np.column_stack(
                        [Ze, logit(xgb_al.predict_proba(Ze)[:, 1]), ae]))
                t0 = time.time()
                p_v2 = fusion_v2(X_b, s_b, X_te, s_te)
                b_v2 = oof(lambda tr, te: fusion_v2(X_b[tr], s_b[tr], X_b[te], s_b[te]))
                record("Fusion-B-v2-k", k, p_v2, t_al + time.time() - t0, benign_oof=b_v2)

            if args.save_scores:
                np.savez_compressed(res_dir / "scores" / f"benign_only_{seed}_{tgt}.npz", **scores)
            pd.DataFrame(rows).to_csv(out_csv, mode="a", index=False, header=not out_csv.exists())

    # ---- 5. Summary table (mean over seeds) ----
    df = pd.read_csv(out_csv)
    tab = (df.groupby(["rule", "target", "budget", "method"])
           [["f1", "balanced_accuracy", "recall", "specificity", "auroc"]].agg(["mean", "std"]).round(4))
    tab.to_csv(res_dir / out_csv.name.replace("_raw", "_summary"))
    q = df[df.rule == f"benign-q{int(BENIGN_Q * 100)}"]
    print(f"\nThreshold rule: benign-q{int(BENIGN_Q * 100)} (mean over seeds)\n",
          q.groupby(["target", "budget", "method"])
          [["f1", "balanced_accuracy", "recall", "specificity", "auroc"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
