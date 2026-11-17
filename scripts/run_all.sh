#!/usr/bin/env bash
# Reproduce every result in order. Each step can also be run on its own.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/prepare_data.py                 # 1 stratified subsample of the raw CSVs
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda_run.ipynb  # 2 EDA figures
python scripts/run_baselines.py --all-seeds    # 3 baselines, in- and cross-dataset
python scripts/summarise_baselines.py          #   paper tables
python scripts/run_fusion.py --all-seeds       # 4 few-shot fusion (GPU)
python scripts/run_benign_only.py --all-seeds  # 5 benign-only target, Fusion-B / v2 (GPU)
python scripts/stats_benign_only.py            # 6 Wilcoxon tests
python scripts/plot_benign_only.py             #   paper figures
python scripts/plot_paper_figures.py           #   remaining result figures
python scripts/eda_shift.py                    #   class distribution + KS shift
python scripts/shap_analysis.py                #   SHAP vs KS
python scripts/run_benign_only.py --save-scores && python scripts/plot_roc_cm.py  #   ROC + confusion matrices
python scripts/make_report.py                  # 7 Word report (results/report.docx)
