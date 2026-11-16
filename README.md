# Cross-dataset generalisation of ML-based intrusion detection on NetFlow v3

Code for the thesis/paper *Do Temporal Features and Tabular Foundation Models
Close the Cross-Network Generalisation Gap? A 4×4 Benchmark on NetFlow v3
Intrusion Detection Datasets*.

Every table and figure in the paper is produced by the scripts in this
repository with fixed random seeds (`configs/default.yaml`).

## Research questions
1. How well do models trained on one network detect attacks on another (all 12 source→target pairs of 4 datasets)?
2. Do the NetFlow v3 temporal features help or hurt transfer compared with the 43 v2 features?
3. With only 0 / 100 / 1,000 labelled flows from the target network, do tabular foundation models (TabPFN, TabICL) beat XGBoost?
4. Which features act as dataset-specific shortcuts (SHAP), and does removing them improve transfer?

## Data
Four NetFlow v3 datasets from the University of Queensland
(<https://staff.itee.uq.edu.au/marius/NIDS_datasets/>):

| Key | Dataset | Flows | Attack % |
|---|---|---|---|
| `toniot` | NF-ToN-IoT-v3 | 27,520,260 | 38.98 |
| `cic2018` | NF-CSE-CIC-IDS2018-v3 | 20,115,529 | 12.93 |
| `botiot` | NF-BoT-IoT-v3 | 16,993,808 | 99.7 |
| `unsw` | NF-UNSW-NB15-v3 | 2,365,424 | 5.4 |

Licence: free for academic research when the dataset papers are cited;
commercial use needs the authors' permission.

**Download:** open the page above, NetFlow V3 tab, and download the four zips from UQ Research Data Manager. Unzip them and copy the four folders (`NF-UNSW-NB15-v3`, `NF-ToN-IoT-v3`, `NF-BoT-IoT-v3`, `NF-CICIDS2018-v3`) into `data/raw/` as they are; `configs/default.yaml` finds the CSV inside each folder.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
On Google Colab: open **`notebooks/00_colab_setup.ipynb`** and run it top to bottom. It runs every step in order (setup, data, EDA, baselines, fusion, benign-only, statistics and figures, Word report), explains each step in Bangla, and writes all results straight to Google Drive so an interrupted session can resume. Raw data is not committed to git.

## Reproduce
```bash
bash scripts/run_all.sh          # every step in order
# or step by step:
python scripts/prepare_data.py   # stratified subsample -> data/processed/*.parquet
jupyter notebook notebooks/01_eda.ipynb
python scripts/run_baselines.py  # RF, XGB, LR, SVM, KNN + soft-voting ensemble; in- and cross-dataset
python scripts/summarise_baselines.py   # paper tables in results/
python scripts/run_fusion.py     # few-shot Fusion vs XGB zero-shot / XGB-k / TabPFN-k (GPU)
python scripts/run_benign_only.py --all-seeds   # benign-only target, proposed Fusion-B v2 (GPU)
python scripts/make_report.py    # results/report.docx
```
**Proposed Fusion model:** XGBoost trained on the source network(s) scores
every target flow; that score is added as a feature, and TabPFN (a pretrained
tabular foundation model) is fitted on only k = 100 or 1,000 labelled target
flows. Default mode is leave-one-network-out (source = the other three).

**Benign-only target setting (main contribution):** the target network has only
k benign labels. `python scripts/run_benign_only.py --all-seeds` compares XGB-src,
IForest-k, XGB-pool-k, Fusion-B-k and the proposed **Fusion-B v2** (TabPFN with
target benign + held-out source attack flows, on features aligned to each
network's own benign traffic, `src/align.py`). Every method is thresholded at the
95th percentile of its cross-fitted scores on the known benign flows.

`run_baselines.py` appends to `results/baselines_raw.csv` and skips finished
runs, so an interrupted Colab session can simply rerun it. Use `--all-seeds`
for the final 5-seed numbers.

Dry run without the real data:
```bash
python scripts/make_toy_data.py tests/toy_raw
python scripts/prepare_data.py --raw-dir tests/toy_raw --out-dir /tmp/toy --n-rows 3000
pytest -q
```

## Layout
```
configs/    experiment settings (seeds, paths, subsample size)
data/       raw/ (downloads, git-ignored) and processed/ (parquet, git-ignored)
notebooks/  analysis notebooks (EDA, figures)
src/        reusable code: data.py, features.py, utils.py (+ models, transfer, few-shot later)
scripts/    one script per experiment step; run_all.sh runs them all
tests/      pytest checks on synthetic data
results/    CSV result tables and figures used in the paper
paper/      LaTeX source
```

## Reproducibility
- Seeds fixed in `configs/default.yaml` (`seed: 42`; repeated runs over 5 seeds).
- Package versions pinned in `requirements.txt`.
- IP addresses, ports and absolute flow timestamps are dropped from all
  feature sets so models cannot memorise the testbed.

## Citation of datasets
- Luay et al., "Temporal Analysis of NetFlow Datasets for Network Intrusion Detection Systems", arXiv:2503.04404 (IEEE Access, 2026).
- Sarhan, Layeghy, Portmann, "Towards a Standard Feature Set for Network Intrusion Detection System Datasets", Mobile Networks and Applications, 2022. doi:10.1007/s11036-021-01843-0
