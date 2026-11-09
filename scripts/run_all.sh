#!/usr/bin/env bash
# Reproduce every result in the paper, in order. Each step can also be run alone.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/prepare_data.py          # week 1: subsample raw CSVs
python scripts/run_baselines.py --all-seeds   # week 3: baselines, in- and cross-dataset
python scripts/summarise_baselines.py
# python scripts/run_fewshot.py         # week 5
# python scripts/run_shap.py            # week 6
