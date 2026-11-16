#!/usr/bin/env bash
# Reproduce every result in the paper, in order. Each step can also be run alone.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/prepare_data.py          # week 1: subsample raw CSVs
python scripts/run_baselines.py --all-seeds   # week 3: baselines, in- and cross-dataset
python scripts/summarise_baselines.py
python scripts/run_fusion.py --all-seeds     # week 5: proposed Fusion model (GPU)
python scripts/run_benign_only.py --all-seeds # benign-only target, proposed Fusion-B v2 (GPU)
python scripts/stats_benign_only.py   # Wilcoxon: Fusion-B v2 vs the others
python scripts/plot_benign_only.py    # paper figures
python scripts/make_report.py                # Word report of all tables (results/report.docx)
# python scripts/run_shap.py            # week 6
