"""Proposed method: source-informed few-shot learning (Fusion).

1. Train XGBoost on labelled flows from SOURCE network(s).
2. On the TARGET network only a few flows are labelled (k = 100 or 1000).
   For every target flow, add the source model's attack score (logit) as an
   extra feature next to the original NetFlow features.
3. Fit TabPFN, a pretrained tabular foundation model that learns from a
   handful of examples in context, on the k labelled target flows and
   predict the rest.

The idea: the source model carries knowledge of what attacks look like; the
few target labels tell TabPFN how far to trust that knowledge in the new
network.
"""
from __future__ import annotations

import numpy as np


def logit(p, eps: float = 1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def make_fewshot_learner(kind: str, seed: int, version: str = "v2"):
    if kind == "rf":  # cheap stand-in so the pipeline can be tested without a GPU
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    if kind == "tabpfn":
        from tabpfn import TabPFNClassifier
        if version == "default":
            return TabPFNClassifier(random_state=seed)
        from tabpfn.constants import ModelVersion
        # v2 weights are openly licensed (no account/token needed)
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2, random_state=seed)
    raise ValueError(kind)


def predict_proba_chunked(model, X, chunk: int = 5000):
    out = [model.predict_proba(X[i:i + chunk])[:, 1] for i in range(0, len(X), chunk)]
    return np.concatenate(out)
