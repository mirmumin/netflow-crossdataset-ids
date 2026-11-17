# Cross-network intrusion detection on NetFlow v3

Code for the thesis/paper on **cross-network generalisation of ML-based
network intrusion detection** on the four NetFlow v3 datasets, and on a
source-informed in-context learner (**Fusion-B**) for a new network that has
only benign labels.

Every table and figure is produced by the scripts in this repository with
fixed random seeds (`configs/default.yaml`).

## Research questions
1. How well do models trained on one network detect attacks on another (all 12 source→target pairs of 4 datasets)?
2. With only k = 100 / 1,000 labelled target flows, does source knowledge help a tabular foundation model (TabPFN) beyond target-only training?
3. When the target network has **only benign labels**, can source attack knowledge plus TabPFN beat target-only anomaly detection and zero-shot transfer? (main contribution)
4. Do the NetFlow v3 temporal features help or hurt transfer compared with the v2 features?

## Main results (seed 42 for steps 3-4, 5 seeds for step 5)
| Setting | Finding |
|---|---|
| In-dataset baselines | F1 84-100% (XGBoost/RF best) |
| Cross-dataset baselines | mean F1 18-33%, AUROC ≈ 0.5: a large generalisation gap |
| Few-shot fusion (k labelled target flows) | Fusion ≈ TabPFN-k ≈ XGB-k; target labels alone suffice |
| Benign-only target (proposed Fusion-B) | mean F1 0.57-0.61 vs 0.25 (IForest) and 0.13 (zero-shot XGBoost); Wilcoxon p < 0.001 |

## Data
Four NetFlow v3 datasets from the University of Queensland
(<https://staff.itee.uq.edu.au/marius/NIDS_datasets/>):

| Key | Dataset | Flows | Attack % |
|---|---|---|---|
| `unsw` | NF-UNSW-NB15-v3 | 2,365,424 | 5.4 |
| `toniot` | NF-ToN-IoT-v3 | 27,520,260 | 38.98 |
| `botiot` | NF-BoT-IoT-v3 | 16,993,808 | 99.7 |
| `cic2018` | NF-CSE-CIC-IDS2018-v3 | 20,115,529 | 12.93 |

Licence: free for academic research when the dataset papers are cited;
commercial use needs the authors' permission. Raw data is not committed.

**Download:** on the page above (NetFlow V3 tab) download the four zips from
UQ Research Data Manager and put them, or the unzipped folders
(`NF-UNSW-NB15-v3`, `NF-ToN-IoT-v3`, `NF-BoT-IoT-v3`, `NF-CICIDS2018-v3`), in
`data/raw/` (locally) or `MyDrive/thesis_data/` (Colab).

## Quick start

### Google Colab (recommended)
Open **`notebooks/00_colab_setup.ipynb`** in Colab with a T4 GPU and run it
top to bottom. It runs Steps 0-7 below in order, explains each step, and
writes all results to Google Drive so an interrupted session can resume.

### Local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_all.sh        # every step in order
```

## Pipeline
| Step | Script | Output (in `results/`) |
|---|---|---|
| 1 Data preparation | `scripts/prepare_data.py` | `data/processed/*.parquet`, `class_summary.csv` |
| 2 EDA (distribution shift) | `notebooks/01_eda.ipynb` | `figures/ks_shift_heatmap.png` |
| 3 Baselines (RF, XGB, SVM, LR, KNN, Ensemble), in- and cross-dataset | `scripts/run_baselines.py`, `scripts/summarise_baselines.py` | `baselines_raw.csv`, `table_models.md`, `table_cross_*.csv` |
| 4 Few-shot fusion | `scripts/run_fusion.py` | `fusion_raw.csv`, `fusion_summary.csv` |
| 5 Benign-only target, Fusion-B / Fusion-B v2 | `scripts/run_benign_only.py --all-seeds` | `benign_only_raw.csv`, `benign_only_summary.csv` |
| 6 Wilcoxon tests and figures | `scripts/stats_benign_only.py`, `scripts/plot_benign_only.py`, `scripts/plot_paper_figures.py` | `stats_benign_only.csv`, `figures/benign_only_*.png`, `figures/fig_*.png` (+ .pdf) |
| 7 Word report | `scripts/make_report.py` | `report.docx` |

All experiment scripts append to their `*_raw.csv` and skip finished runs, so
they can simply be restarted. `--all-seeds` uses the 5 seeds in the config.

### Proposed method: Fusion-B
The target network has k labelled **benign** flows and no labelled attacks;
the other three networks (leave-one-network-out) supply labelled attacks.
TabPFN is given an in-context training set of the k target benign flows
(label 0) and source attack flows (label 1), with two extra features: the
score of an XGBoost trained on the source networks and an Isolation Forest
anomaly score fitted on the target benign flows. Fusion-B v2 first maps each
network's features to the quantiles of its own benign traffic (`src/align.py`).

Leakage and fairness rules: the source attack flows in the context are held
out from the source XGBoost; anomaly scores of the context flows are
cross-fitted; every method is thresholded at the 95th percentile of its
cross-fitted scores on the known benign flows (only benign labels are used).

## Repository layout
```
configs/    default.yaml: seeds, paths, subsample sizes, model settings
data/       raw/ (downloads) and processed/ (parquet); both git-ignored
notebooks/  00_colab_setup.ipynb (full pipeline on Colab), 01_eda.ipynb
src/        data.py, features.py, splits.py, models.py, evaluate.py, fusion.py, align.py, utils.py
scripts/    one script per pipeline step; run_all.sh runs them all
tests/      pytest checks on synthetic data (python scripts/make_toy_data.py)
docs/figures/  method diagram (Figure 2) as SVG/PNG/PDF and the script that draws it
results/    result tables and figures (created by the scripts)
```

## Tests
```bash
pytest -q        # runs the whole pipeline on small synthetic data (TabPFN replaced by RF)
```

## Reproducibility
- Seeds fixed in `configs/default.yaml` (`seed: 42`; `seeds: 42-46` for repeated runs).
- Package versions pinned in `requirements.txt`; TabPFN uses the openly licensed v2 weights.
- IP addresses, ports and absolute flow timestamps are dropped from all
  feature sets so models cannot memorise the testbed.

## Citation of datasets
- Luay et al., "Temporal Analysis of NetFlow Datasets for Network Intrusion Detection Systems", arXiv:2503.04404 (IEEE Access, 2026).
- Sarhan, Layeghy, Portmann, "Towards a Standard Feature Set for Network Intrusion Detection System Datasets", Mobile Networks and Applications, 2022. doi:10.1007/s11036-021-01843-0
