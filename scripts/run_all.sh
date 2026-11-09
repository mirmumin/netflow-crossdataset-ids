#!/usr/bin/env bash
# Reproduce every result in the paper, in order. Each step can also be run alone.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/prepare_data.py          # week 1: subsample raw CSVs
# python scripts/run_baselines.py       # week 3 (to be added)
# python scripts/run_transfer.py        # week 3-4 (4x4 cross-dataset)
# python scripts/run_fewshot.py         # week 5
# python scripts/run_shap.py            # week 6
