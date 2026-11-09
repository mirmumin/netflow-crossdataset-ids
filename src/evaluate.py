"""Binary metrics (attack = positive class)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def binary_metrics(y_true, p_attack, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(p_attack) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),          # detection rate
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "fpr": fp / (fp + tn) if (fp + tn) else np.nan,                   # false alarm rate
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    out["auroc"] = roc_auc_score(y_true, p_attack) if len(np.unique(y_true)) == 2 else np.nan
    return out
