"""Baseline classifiers and the soft-voting ensemble.

All models see the same preprocessing: log1p on the (non-negative, heavy
tailed) NetFlow counters, then standardisation fitted on the SOURCE training
data only. Fitting the scaler on the target would leak target information.

SVM and KNN do not scale to hundreds of thousands of rows, so each model has
its own cap on training rows (config: baselines.max_train). The cap is applied
with a stratified sample using the run's seed.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC


def _log1p_clip(x):
    return np.log1p(np.clip(x, 0, None))


def preprocessor():
    return [FunctionTransformer(_log1p_clip, feature_names_out="one-to-one"), StandardScaler()]


def make_model(name: str, seed: int):
    if name == "RF":
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed,
                                     class_weight="balanced_subsample")
    elif name == "XGB":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                            n_jobs=-1, random_state=seed, eval_metric="logloss")
    elif name == "LR":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    elif name == "SVM":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                  class_weight="balanced", random_state=seed)
    elif name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    else:
        raise ValueError(f"unknown model {name}")
    return make_pipeline(*preprocessor(), clf)


class SoftVotingEnsemble:
    """Average of the attack probabilities of already-fitted members.

    Members are fitted separately (each with its own row cap), so sklearn's
    VotingClassifier, which refits everything on one dataset, does not fit here.
    """

    def __init__(self, members: dict):
        self.members = members

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.members.values()], axis=0)
        return np.column_stack([1 - p, p])
