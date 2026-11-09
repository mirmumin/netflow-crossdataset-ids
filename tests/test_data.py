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
