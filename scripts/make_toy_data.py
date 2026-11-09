"""Create small synthetic CSVs with the NetFlow v3 column layout.

Used only by the tests and for a dry run of the pipeline without the real
multi-GB downloads. Values are random; never report results on them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import V2_FEATURES  # noqa: E402

V3_EXTRA = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS",
            "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG",
            "SRC_TO_DST_IAT_STDDEV", "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX",
            "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV"]


def make(path: Path, n: int, attacks: dict[str, float], seed: int) -> None:
    rng = np.random.default_rng(seed)
    cols = V2_FEATURES[:]
    cols[13:13] = V3_EXTRA[:2]  # timestamps sit next to duration in v3
    cols += V3_EXTRA[2:]
    df = pd.DataFrame(rng.integers(0, 1000, size=(n, len(cols))), columns=cols)
    names = list(attacks)
    df["Attack"] = rng.choice(names, size=n, p=list(attacks.values()))
    df["Label"] = (df["Attack"] != "Benign").astype(int)
    df.to_csv(path, index=False)


def main(out: str = "tests/toy_raw") -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "NF-UNSW-NB15-v3.csv": {"Benign": 0.94, "Exploits": 0.04, "Worms": 0.02},
        "NF-ToN-IoT-v3.csv": {"Benign": 0.6, "scanning": 0.3, "xss": 0.1},
        "NF-BoT-IoT-v3.csv": {"Benign": 0.01, "DDoS": 0.6, "DoS": 0.39},
        "NF-CSE-CIC-IDS2018-v3.csv": {"Benign": 0.87, "DDOS": 0.1, "Infilteration": 0.03},
    }
    for i, (name, p) in enumerate(specs.items()):
        make(out_dir / name, 5000, p, seed=i)
    print(f"toy CSVs written to {out_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:])
