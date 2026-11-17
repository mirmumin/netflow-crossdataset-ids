import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import class_fractions  # noqa: E402
from src.features import feature_sets  # noqa: E402


def test_class_fractions_keeps_small_classes():
    counts = pd.Series({"Benign": 1_000_000, "DoS": 50_000, "Worms": 150})
    f = class_fractions(counts, n_rows=100_000, min_per_class=2_000)
    assert f["Worms"] == 1.0
    kept = sum(counts[k] * f[k] for k in counts.index)
    assert 95_000 < kept < 105_000


def test_pipeline_on_toy_data(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    subprocess.run([sys.executable, "scripts/make_toy_data.py", str(raw)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/prepare_data.py", "--raw-dir", str(raw),
                    "--out-dir", str(out), "--n-rows", "2000"], cwd=ROOT, check=True)
    for key in ["unsw", "toniot", "botiot", "cic2018"]:
        df = pd.read_parquet(out / f"{key}.parquet")
        # min_per_class=2000 floor can push small toy sets above n_rows
        assert 1500 < len(df) <= 5000
        fs = feature_sets(df.columns)
        assert len(fs["v2"]) == 39            # 43 minus 4 identifiers
        assert len(fs["v3"]) == 39 + 8        # + 8 temporal (2 timestamps dropped)
        assert "IPV4_SRC_ADDR" not in fs["v3"]
    assert (out / "class_summary.csv").exists()


def test_subsample_is_deterministic(tmp_path):
    raw = tmp_path / "raw"
    subprocess.run([sys.executable, "scripts/make_toy_data.py", str(raw)], cwd=ROOT, check=True)
    outs = []
    for i in range(2):
        out = tmp_path / f"out{i}"
        subprocess.run([sys.executable, "scripts/prepare_data.py", "--raw-dir", str(raw),
                        "--out-dir", str(out), "--n-rows", "2000", "--only", "unsw"],
                       cwd=ROOT, check=True)
        outs.append(pd.read_parquet(out / "unsw.parquet"))
    pd.testing.assert_frame_equal(outs[0], outs[1])


def test_min_benign_keeps_benign_in_attack_heavy_data():
    counts = pd.Series({"DDoS": 10_000_000, "DoS": 6_900_000, "Benign": 50_000})
    f = class_fractions(counts, n_rows=500_000, min_per_class=2_000, min_benign=100_000)
    assert f["Benign"] == 1.0          # all 50k benign kept


def test_baselines_and_summary_on_toy_data(tmp_path):
    raw, proc, res = tmp_path / "raw", tmp_path / "proc", tmp_path / "res"
    subprocess.run([sys.executable, "scripts/make_toy_data.py", str(raw)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/prepare_data.py", "--raw-dir", str(raw),
                    "--out-dir", str(proc), "--n-rows", "2000"], cwd=ROOT, check=True)
    cmd = [sys.executable, "scripts/run_baselines.py", "--processed-dir", str(proc),
           "--results-dir", str(res), "--models", "LR", "KNN", "--feature-sets", "v3"]
    subprocess.run(cmd, cwd=ROOT, check=True)
    df = pd.read_csv(res / "baselines_raw.csv")
    assert len(df) == 4 * 4 * 2                     # no ENS: not all members requested
    cmd = cmd[:-4] + ["--models", "RF", "XGB", "LR", "SVM", "KNN", "--feature-sets", "v3"]
    subprocess.run(cmd, cwd=ROOT, check=True)       # LR/KNN refitted only for ENS
    df = pd.read_csv(res / "baselines_raw.csv")
    assert len(df) == 4 * 4 * 6
    assert (df.loc[df.model == "ENS", "members"] == "RF+XGB+LR+SVM+KNN").all()
    assert {"accuracy", "precision", "recall", "f1", "specificity"} <= set(df.columns)
    subprocess.run(cmd, cwd=ROOT, check=True)       # resume: nothing re-run
    assert len(pd.read_csv(res / "baselines_raw.csv")) == len(df)
    subprocess.run([sys.executable, "scripts/summarise_baselines.py", str(res)], cwd=ROOT, check=True)
    assert (res / "table_cross_f1_v3.csv").exists()


def test_fusion_pipeline_with_rf_standin(tmp_path):
    raw, proc, res = tmp_path / "raw", tmp_path / "proc", tmp_path / "res"
    subprocess.run([sys.executable, "scripts/make_toy_data.py", str(raw)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/prepare_data.py", "--raw-dir", str(raw),
                    "--out-dir", str(proc), "--n-rows", "2000"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/run_fusion.py", "--processed-dir", str(proc),
                    "--results-dir", str(res), "--learner", "rf", "--budgets", "100",
                    "--targets", "unsw"], cwd=ROOT, check=True)
    df = pd.read_csv(res / "fusion_raw_rf.csv")
    assert set(df["method"]) == {"XGB-src", "XGB-k", "rf-k", "Fusion-k"}
    assert (df["source"] == "toniot+botiot+cic2018").all()


def test_report_builds(tmp_path):
    raw, proc, res = tmp_path / "raw", tmp_path / "proc", tmp_path / "res"
    subprocess.run([sys.executable, "scripts/make_toy_data.py", str(raw)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/prepare_data.py", "--raw-dir", str(raw),
                    "--out-dir", str(proc), "--n-rows", "2000"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/run_baselines.py", "--processed-dir", str(proc),
                    "--results-dir", str(res), "--models", "LR", "KNN", "--feature-sets", "v3"],
                   cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/make_report.py", str(res)], cwd=ROOT, check=True)
    assert (res / "report.docx").stat().st_size > 10_000


def test_benign_only_pipeline_with_rf_standin(tmp_path):
    raw, proc, res = tmp_path / "raw", tmp_path / "proc", tmp_path / "res"
    subprocess.run([sys.executable, "scripts/make_toy_data.py", str(raw)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/prepare_data.py", "--raw-dir", str(raw),
                    "--out-dir", str(proc), "--n-rows", "2000"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/run_benign_only.py", "--processed-dir", str(proc),
                    "--results-dir", str(res), "--learner", "rf", "--budgets", "100",
                    "--targets", "toniot"], cwd=ROOT, check=True)
    df = pd.read_csv(res / "benign_only_raw_rf.csv")
    assert set(df["method"]) == {"XGB-src", "IForest-k", "XGB-pool-k", "Fusion-B-k",
                                 "XGB-src-BQN-k", "Fusion-B-v2-k"}


def test_benign_quantile_aligner_centres_benign():
    import numpy as np
    from src.align import BenignQuantileAligner
    rng = np.random.default_rng(0)
    ben = rng.normal(10, 2, size=(500, 3))
    al = BenignQuantileAligner().fit(ben)
    z = al.transform(np.array([[10, 10, 10], [100, 100, 100]]))
    assert np.allclose(z[0], 0.5, atol=0.1)   # typical benign value -> middle
    assert np.allclose(z[1], 1.0)             # far above all benign -> 1


def test_iforest_threshold_gives_low_false_alarm():
    import importlib.util
    import numpy as np
    spec = importlib.util.spec_from_file_location("rbo", ROOT / "scripts" / "run_benign_only.py")
    rbo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rbo)
    rng = np.random.default_rng(0)
    anom = rbo.BenignAnomaly(rng.normal(size=(300, 4)), seed=0)
    s = anom.score(rng.normal(size=(5000, 4)))     # unseen benign flows
    thr = np.quantile(anom.score_train(), rbo.BENIGN_Q)
    assert (s >= thr).mean() < 0.12
    tr = anom.score_train()
    assert 0.3 < tr.mean() < 0.7                   # cross-fitted scores are not all low


def test_eda_shap_and_roc_scripts_on_toy_data(tmp_path):
    raw, proc, res = tmp_path / "raw", tmp_path / "proc", tmp_path / "res"
    subprocess.run([sys.executable, "scripts/make_toy_data.py", str(raw)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/prepare_data.py", "--raw-dir", str(raw),
                    "--out-dir", str(proc), "--n-rows", "2000"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/eda_shift.py", str(res), "--processed-dir", str(proc)],
                   cwd=ROOT, check=True)
    assert (res / "ks_shift_benign.csv").exists()
    subprocess.run([sys.executable, "scripts/shap_analysis.py", str(res), "--processed-dir", str(proc),
                    "--n-explain", "200"], cwd=ROOT, check=True)
    imp = pd.read_csv(res / "shap_importance.csv", index_col=0)
    assert abs(imp.drop(columns="mean").sum().sum() - 4) < 1e-6   # each network normalised to 1
    assert (res / "shap_vs_ks.csv").exists()
    subprocess.run([sys.executable, "scripts/run_benign_only.py", "--processed-dir", str(proc),
                    "--results-dir", str(res), "--learner", "rf", "--budgets", "100",
                    "--targets", "toniot", "unsw", "--save-scores"], cwd=ROOT, check=True)
    assert not (res / "benign_only_raw_rf.csv").exists()          # main results file untouched
    assert (res / "scores" / "benign_only_42_toniot.npz").exists()
    subprocess.run([sys.executable, "scripts/plot_roc_cm.py", str(res)], cwd=ROOT, check=True)
    for name in ["fig_class_distribution", "fig_ks_shift", "fig_shap_vs_ks", "fig_roc_k100", "fig_cm_k100"]:
        assert (res / "figures" / f"{name}.png").exists()
