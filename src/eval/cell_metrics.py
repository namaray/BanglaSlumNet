"""
Cell-level benchmark metrics (numpy/sklearn; the torch metrics in metrics.py are for
pixel segmentation). Labels: 1 slum, 0 non-slum.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import roc_auc_score


def binary_metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(prob) >= threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    recall = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    return {
        "auc": roc_auc_score(y, prob) if 0 < y.sum() < len(y) else np.nan,
        "balanced_acc": np.nanmean([recall, spec]),
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else np.nan,
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
        "pred_pos_rate": float(pred.mean()),
        "n_slum": int(y.sum()),
        "n_nonslum": int((~y).sum()),
    }


def stratum_fpr(y: np.ndarray, prob: np.ndarray, stratum: np.ndarray,
                threshold: float = 0.5) -> Dict[str, Optional[float]]:
    """False-positive rate on the non-slum cells of a stratum (e.g. dense formal areas)."""
    neg = (np.asarray(y) == 0) & np.asarray(stratum, dtype=bool)
    n = int(neg.sum())
    return {"fpr": float((np.asarray(prob)[neg] >= threshold).mean()) if n else np.nan, "n": n}
