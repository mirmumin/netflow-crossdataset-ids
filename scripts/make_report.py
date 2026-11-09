"""Build a Word report (results/report.docx) from whatever results exist.

Sections are added only for result files that are present, so it can be run
after the baselines alone, or again after the fusion run.

Usage:
    python scripts/make_report.py                     # reads results/
    python scripts/make_report.py /path/to/results    # e.g. the Drive copy
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from docx import Document  # noqa: E402
from docx.shared import Pt  # noqa: E402

from src.utils import load_config, resolve  # noqa: E402

NAMES = {"unsw": "NF-UNSW-NB15-v3", "toniot": "NF-ToN-IoT-v3",
         "botiot": "NF-BoT-IoT-v3", "cic2018": "NF-CSE-CIC-IDS2018-v3"}
MODEL_ORDER = ["RF", "XGB", "SVM", "LR", "KNN", "ENS"]
MODEL_NAMES = {"XGB": "XGBoost", "ENS": "Ensemble (soft voting)"}
FUSION_ORDER = ["XGB-src", "XGB-k", "TabPFN-k", "Fusion-k"]
FUSION_NAMES = {"XGB-src": "XGBoost, source only (zero-shot)",
                "XGB-k": "XGBoost, target labels only",
                "TabPFN-k": "TabPFN, target labels only",
                "Fusion-k": "Fusion (proposed)"}


def add_table(doc, df: pd.DataFrame, index_name: str = "", bold_max_cols=None, fmt="{:.2f}"):
    t = doc.add_table(rows=1, cols=len(df.columns) + 1)
    t.style = "Light Grid Accent 1"
    hdr = t.rows[0].cells
    hdr[0].text = index_name
    for j, c in enumerate(df.columns):
        hdr[j + 1].text = str(c)
    best = {c: df[c].max() for c in (bold_max_cols or [])}
    for idx, row in df.iterrows():
        cells = t.add_row().cells
        cells[0].text = str(idx)
        for j, c in enumerate(df.columns):
            v = row[c]
            txt = fmt.format(v) if isinstance(v, (int, float)) and pd.notna(v) else ("–" if pd.isna(v) else str(v))
            run = cells[j + 1].paragraphs[0].add_run(txt)
            if c in best and pd.notna(v) and v == best[c]:
                run.bold = True
    doc.add_paragraph()


def main(results_dir: str | None = None) -> None:
    cfg = load_config()
    res = resolve(results_dir or cfg["paths"]["results_dir"])
    doc = Document()
    doc.styles["Normal"].font.size = Pt(10)
    doc.add_heading("Cross-network intrusion detection on NetFlow v3: results", 0)
    doc.add_paragraph(f"Generated {date.today():%d %B %Y} from {res.name}/ by scripts/make_report.py. "
                      "Code: github.com/mirmumin/netflow-crossdataset-ids")
    doc.add_heading("Setup", 1)
    for line in [
        "Data: four UQ NetFlow v3 datasets, stratified subsample of ~500,000 flows each "
        "(all attack classes kept; at least 100,000 benign flows).",
        "Features: 53 NetFlow v3 features minus IP addresses, ports and absolute timestamps "
        "(47 used); infinite values capped, then log(1 + x).",
        "Split: 80% train / 20% test per dataset, stratified by attack class. Task: benign vs attack.",
        "In-dataset = trained and tested on the same network. Cross-dataset = trained on one "
        "network, tested on another network's test split.",
        f"Seeds: {cfg['seed']} (single seed unless stated). Attack is the positive class.",
    ]:
        doc.add_paragraph(line, style="List Bullet")

    raw = res / "baselines_raw.csv"
    if raw.exists():
        df = pd.read_csv(raw)
        for fs in sorted(df.feature_set.unique()):
            d = df[df.feature_set == fs]
            doc.add_heading(f"1. Classical models, in-dataset ({fs} features)", 1)
            doc.add_paragraph("Mean over seeds, in %. Best value per column in bold.")
            ind = d[d.source == d.target]
            for src in [k for k in NAMES if k in set(ind.source)]:
                g = ind[ind.source == src]
                t = g.groupby("model")[["accuracy", "precision", "recall", "f1", "specificity"]].mean() * 100
                t = t.reindex([m for m in MODEL_ORDER if m in t.index])
                t.index = [MODEL_NAMES.get(m, m) for m in t.index]
                t.columns = ["Accuracy", "Precision", "Recall", "F1", "Specificity"]
                doc.add_heading(NAMES[src], 2)
                add_table(doc, t, "Model", bold_max_cols=list(t.columns))

            cross = d[d.source != d.target]
            if len(cross):
                doc.add_heading(f"2. Cross-dataset generalisation ({fs} features)", 1)
                doc.add_paragraph("Mean over all source→target pairs (12 when all four datasets are done), in %.")
                t = cross.groupby("model")[["accuracy", "precision", "recall", "f1", "specificity",
                                            "balanced_accuracy"]].mean() * 100
                t = t.reindex([m for m in MODEL_ORDER if m in t.index])
                t.index = [MODEL_NAMES.get(m, m) for m in t.index]
                t.columns = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "Bal. accuracy"]
                add_table(doc, t, "Model", bold_max_cols=list(t.columns))
                doc.add_paragraph("F1 (%) for every pair. Rows: training network; columns: test network; "
                                  "diagonal = in-dataset.")
                for m in [m for m in MODEL_ORDER if m in set(d.model)]:
                    mat = (d[d.model == m].groupby(["source", "target"])["f1"].mean() * 100).unstack("target")
                    order = [k for k in NAMES if k in mat.index or k in mat.columns]
                    mat = mat.reindex(index=[k for k in order if k in mat.index],
                                      columns=[k for k in order if k in mat.columns])
                    mat.index = [NAMES[k] for k in mat.index]
                    mat.columns = [NAMES[k].replace("-v3", "") for k in mat.columns]
                    doc.add_heading(MODEL_NAMES.get(m, m), 3)
                    add_table(doc, mat, "Train \\ Test", fmt="{:.1f}")

    fus = res / "fusion_raw.csv"
    if fus.exists():
        f = pd.read_csv(fus)
        doc.add_heading("3. Proposed Fusion model vs few-shot and zero-shot baselines", 1)
        doc.add_paragraph(
            "Target network has only k labelled flows. Source = the other three networks "
            "(leave-one-network-out). Fusion: XGBoost trained on the source scores each target "
            "flow; the score is added as a feature and TabPFN is fitted on the k labelled target "
            "flows. All methods are tested on the same target test flows. Values in %.")
        for tgt in [k for k in NAMES if k in set(f.target)]:
            g = f[f.target == tgt]
            t = g.groupby(["budget", "method"])[["f1", "balanced_accuracy", "recall", "specificity"]].mean() * 100
            t = t.reset_index()
            t["order"] = t.method.map({m: i for i, m in enumerate(FUSION_ORDER)})
            t = t.sort_values(["budget", "order"])
            t.index = [f"{FUSION_NAMES.get(m, m)}" + (f", k={b}" if b else "") for b, m in zip(t.budget, t.method)]
            t = t[["f1", "balanced_accuracy", "recall", "specificity"]]
            t.columns = ["F1", "Bal. accuracy", "Recall", "Specificity"]
            doc.add_heading(f"Target: {NAMES[tgt]}", 2)
            add_table(doc, t, "Method", bold_max_cols=["F1", "Bal. accuracy"])
        n_seeds = f.seed.nunique()
        doc.add_paragraph(f"Seeds used: {n_seeds}. "
                          + ("Single-seed demo; final paper numbers need --all-seeds." if n_seeds == 1 else ""))

    out = res / "report.docx"
    doc.save(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
