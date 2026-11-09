"""Feature groups of the UQ NetFlow datasets.

NetFlow v2 has 43 features (Sarhan, Layeghy & Portmann, MONET 2022).
NetFlow v3 adds temporal features (Luay et al., arXiv:2503.04404).
The v3-only columns are derived at run time as "columns present in the file
that are not v2 features or labels", so a wrong guess here cannot silently
drop or add columns: check_schema() reports exactly what was found.
"""
from __future__ import annotations

import pandas as pd

LABEL_COL = "Label"     # 0 = benign, 1 = attack
ATTACK_COL = "Attack"   # attack class name ("Benign" for benign flows)

# Identifiers: they describe the testbed, not the behaviour, and let a model
# memorise hosts. Dropped for every experiment (standard practice; see
# Layeghy & Portmann 2023).
ID_COLS = ["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT"]

V2_FEATURES = [
    "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT",
    "PROTOCOL", "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
]
assert len(V2_FEATURES) == 43

# Absolute timestamps are dropped like IDs: they encode *when* the capture
# ran, which differs per dataset and is a pure shortcut.
TIMESTAMP_HINTS = ("FLOW_START", "FLOW_END")


def v3_extra_columns(columns) -> list[str]:
    known = set(V2_FEATURES) | {LABEL_COL, ATTACK_COL}
    return [c for c in columns if c not in known and c != "Dataset"]


def feature_sets(columns) -> dict[str, list[str]]:
    """Return the feature lists used in the ablation.

    v2:  43 v2 features minus identifiers
    v3:  v2 + v3 temporal features, minus identifiers and absolute timestamps
    """
    extra = v3_extra_columns(columns)
    temporal = [c for c in extra if not c.startswith(TIMESTAMP_HINTS)]
    v2 = [c for c in V2_FEATURES if c in columns and c not in ID_COLS]
    return {"v2": v2, "v3": v2 + temporal}


def check_schema(df: pd.DataFrame, name: str = "") -> dict:
    """Report which expected columns exist; raise if labels are missing."""
    cols = list(df.columns)
    missing_v2 = [c for c in V2_FEATURES if c not in cols]
    for c in (LABEL_COL, ATTACK_COL):
        if c not in cols:
            raise ValueError(f"{name}: label column '{c}' not found; got {cols[:10]}...")
    extra = v3_extra_columns(cols)
    return {"dataset": name, "n_cols": len(cols), "missing_v2": missing_v2,
            "v3_extra": extra, "n_v3_extra": len(extra)}
