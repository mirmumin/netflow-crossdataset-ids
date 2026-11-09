"""Step 1: stratified subsample of each raw v3 CSV -> data/processed/<key>.parquet

Usage:
    python scripts/prepare_data.py                 # all datasets in config
    python scripts/prepare_data.py --only unsw     # one dataset
    python scripts/prepare_data.py --raw-dir tests/toy_raw --out-dir /tmp/x  # toy run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data import stratified_subsample, summarise  # noqa: E402
from src.features import check_schema  # noqa: E402
from src.utils import load_config, resolve, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="dataset keys")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-rows", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    set_seed(cfg["seed"])
    raw_dir = resolve(args.raw_dir or cfg["paths"]["raw_dir"])
    out_dir = resolve(args.out_dir or cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = cfg["subsample"]
    n_rows = args.n_rows or sub["n_rows"]

    summaries, schemas = [], []
    for key, d in cfg["datasets"].items():
        if args.only and key not in args.only:
            continue
        csv = raw_dir / d["file"]
        if not csv.exists():
            print(f"[skip] {csv} not found. Download it first (see README).")
            continue
        print(f"[{key}] subsampling {csv.name} -> {n_rows} rows ...")
        df = stratified_subsample(csv, n_rows, sub["min_per_class"],
                                  sub["chunksize"], cfg["seed"])
        schemas.append(check_schema(df, d["name"]))
        df.to_parquet(out_dir / f"{key}.parquet", index=False)
        summaries.append(summarise(df, d["name"]))
        print(f"[{key}] kept {len(df):,} rows, attack rate {df['Label'].mean():.3f}")

    if summaries:
        pd.concat(summaries).to_csv(out_dir / "class_summary.csv", index=False)
        (out_dir / "schema_report.json").write_text(json.dumps(schemas, indent=2))
        print(f"Wrote class_summary.csv and schema_report.json to {out_dir}")


if __name__ == "__main__":
    main()
