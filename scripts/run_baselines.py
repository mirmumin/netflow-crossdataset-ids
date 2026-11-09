"""Step 2: classical baselines, in-dataset and 4x4 cross-dataset.

For each seed, feature set (v2 / v3) and SOURCE dataset:
  * split every dataset 80/20 (stratified on attack class, same seed)
  * train each model on the source training split
  * evaluate on the test split of EVERY dataset
      source == target -> in-dataset result
      source != target -> cross-dataset (transfer) result
Rows are appended to results/baselines_raw.csv as they finish, so an
interrupted Colab run can be resumed (finished rows are skipped).

Usage:
    python scripts/run_baselines.py                       # first seed only
    python scripts/run_baselines.py --all-seeds           # 5 seeds (final numbers)
    python scripts/run_baselines.py --models RF XGB --feature-sets v3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.evaluate import binary_metrics  # noqa: E402
from src.features import LABEL_COL, feature_sets  # noqa: E402
from src.models import SoftVotingEnsemble, make_model  # noqa: E402
from src.splits import cap, split  # noqa: E402
from src.utils import load_config, resolve, set_seed  # noqa: E402

KEYS = ["seed", "feature_set", "source", "target", "model"]


def append_rows(path: Path, rows: list[dict]) -> None:
    """Append rows, matching the column order already in the file."""
    df = pd.DataFrame(rows)
    if path.exists():
        header = pd.read_csv(path, nrows=0).columns.tolist()
        df = df.reindex(columns=header + [c for c in df.columns if c not in header])
        if len(df.columns) > len(header):        # new column: rewrite whole file
            pd.concat([pd.read_csv(path), df]).to_csv(path, index=False)
            return
    df.to_csv(path, mode="a", index=False, header=not path.exists())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--feature-sets", nargs="*", default=None)
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--no-ensemble", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    bl = cfg["baselines"]
    proc = resolve(args.processed_dir or cfg["paths"]["processed_dir"])
    res_dir = resolve(args.results_dir or cfg["paths"]["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)
    out_csv = res_dir / "baselines_raw.csv"

    keys = [k for k in cfg["datasets"] if (proc / f"{k}.parquet").exists()]
    if not keys:
        raise SystemExit(f"No processed parquet files in {proc}. Run prepare_data.py first.")
    data = {k: pd.read_parquet(proc / f"{k}.parquet") for k in keys}
    models = args.models or bl["models"]
    fsets = args.feature_sets or bl["feature_sets"]
    sources = args.sources or keys
    seeds = cfg["seeds"] if args.all_seeds else [cfg["seed"]]
    # the ensemble is only built from its full member list, so "ENS" always
    # means the same model; partial ensembles from earlier runs are dropped
    ens_members = list(bl["ensemble"])
    ens_tag = "+".join(ens_members)
    use_ens = not args.no_ensemble and all(m in models for m in ens_members)

    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        rewrite = "members" not in prev.columns   # file from an older version
        if rewrite:
            prev["members"] = ""
        stale = (prev["model"] == "ENS") & (prev["members"].fillna("") != ens_tag)
        if stale.any():
            print(f"Dropping {int(stale.sum())} ensemble rows built from a different member list.")
            prev = prev[~stale]
        if rewrite or stale.any():
            prev.to_csv(out_csv, index=False)
        done = set(map(tuple, prev[KEYS].astype(str).values))

    # common feature list across all datasets
    cols = set.intersection(*(set(d.columns) for d in data.values()))
    fs_all = feature_sets([c for c in next(iter(data.values())).columns if c in cols])

    for seed in seeds:
        set_seed(seed)
        splits = {k: split(d, cfg["split"]["test_size"], seed) for k, d in data.items()}
        tests = {k: cap(te, bl["max_test"], seed) for k, (_, te) in splits.items()}
        for fs in fsets:
            feats = fs_all[fs]
            for src in sources:
                tr = splits[src][0]
                fitted = {}
                ens_pending = use_ens and any((str(seed), fs, src, t, "ENS") not in done for t in keys)
                for m in models + (["ENS"] if use_ens else []):
                    todo = [t for t in keys if (str(seed), fs, src, t, m) not in done]
                    # an ensemble member whose own rows are done is still refitted
                    # (not re-recorded) when the ensemble itself is pending
                    refit_only = not todo and ens_pending and m in ens_members
                    if not todo and not refit_only:
                        continue
                    t0 = time.time()
                    if m == "ENS":
                        model = SoftVotingEnsemble({e: fitted[e] for e in ens_members})
                        fit_s = sum(fitted[e].fit_seconds_ for e in ens_members)
                    else:
                        trc = cap(tr, bl["max_train"].get(m), seed)
                        model = make_model(m, seed)
                        model.fit(trc[feats].to_numpy("float32"), trc[LABEL_COL].to_numpy())
                        fit_s = time.time() - t0
                        model.fit_seconds_ = fit_s
                        fitted[m] = model
                        if refit_only:
                            print(f"seed={seed} {fs} {src:>8} {m:>4} refitted for ENS ({fit_s:.1f}s)")
                            continue
                    rows = []
                    for tgt in todo:
                        te = tests[tgt]
                        p = model.predict_proba(te[feats].to_numpy("float32"))[:, 1]
                        r = {"seed": seed, "feature_set": fs, "source": src, "target": tgt,
                             "model": m, "n_train": len(tr) if m == "ENS" else len(trc),
                             "n_test": len(te), "fit_seconds": round(fit_s, 2),
                             "members": ens_tag if m == "ENS" else ""}
                        r.update(binary_metrics(te[LABEL_COL].to_numpy(), p))
                        rows.append(r)
                    append_rows(out_csv, rows)
                    f1s = " ".join(f"{x['target']}={x['f1']:.3f}" for x in rows)
                    print(f"seed={seed} {fs} {src:>8} {m:>4} fit={fit_s:6.1f}s  F1: {f1s}")
                # fitted models of models not re-run this time are unavailable
    print(f"Results in {out_csv}. Next: python scripts/summarise_baselines.py")


if __name__ == "__main__":
    main()
