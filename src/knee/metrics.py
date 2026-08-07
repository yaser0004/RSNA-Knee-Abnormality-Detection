import numpy as np
from sklearn.metrics import roc_auc_score


def per_label_auc(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-column ROC AUC. NaN targets (e.g. a lexical rule with no match for
    that study) are excluded from that column's score, not treated as a class.
    A column with fewer than two classes among its non-NaN rows gets NaN
    instead of raising, since AUC is undefined there."""
    n_labels = y_true.shape[1]
    aucs = np.full(n_labels, np.nan)
    for i in range(n_labels):
        labeled = ~np.isnan(y_true[:, i])
        if len(np.unique(y_true[labeled, i])) < 2:
            continue
        aucs[i] = roc_auc_score(y_true[labeled, i], y_pred[labeled, i])
    return aucs


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged ROC AUC, excluding labels with only one class present."""
    aucs = per_label_auc(y_true, y_pred)
    return float(np.nanmean(aucs))
