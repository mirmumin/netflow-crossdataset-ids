"""Benign-anchored feature alignment between networks (used by Fusion-B v2).

Problem: the same NetFlow feature has a different scale in every network
(e.g. typical flow duration or byte counts differ), so a model trained on
source networks sees target flows as "out of distribution" (zero-shot AUROC ~0.5).

Idea: describe every flow relative to the NORMAL traffic of its own network.
For each network we fit a per-feature quantile map on that network's benign
flows only, and replace each feature value by its quantile in [0, 1]:
  0.5  = a typical value for benign traffic in this network
  1.0  = larger than every benign flow seen in this network
Source networks use their labelled benign flows; the target network uses only
its k labelled benign flows. Using benign flows (not all flows) makes the map
independent of how many attacks a network happens to contain.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import QuantileTransformer


class BenignQuantileAligner:
    def __init__(self, max_fit: int = 100_000, seed: int = 0):
        self.max_fit = max_fit
        self.seed = seed

    def fit(self, X_benign: np.ndarray) -> "BenignQuantileAligner":
        X_benign = np.asarray(X_benign)
        if len(X_benign) > self.max_fit:
            idx = np.random.default_rng(self.seed).choice(len(X_benign), self.max_fit, replace=False)
            X_benign = X_benign[idx]
        self.qt_ = QuantileTransformer(n_quantiles=min(1000, len(X_benign)),
                                       output_distribution="uniform",
                                       subsample=10**9, random_state=self.seed)
        self.qt_.fit(X_benign)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.qt_.transform(np.asarray(X)).astype("float32")
