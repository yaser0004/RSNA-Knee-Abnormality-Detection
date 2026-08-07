import numpy as np
from sklearn.metrics import roc_auc_score


def per_label_auc(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-column ROC AUC. A column with only one class present gets NaN
    instead of raising, since AUC is undefined there."""
    n_labels = y_true.shape[1]
    aucs = np.full(n_labels, np.nan)
    for i in range(n_labels):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        aucs[i] = roc_auc_score(y_true[:, i], y_pred[:, i])
    return aucs


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged ROC AUC, excluding labels with only one class present."""
    aucs = per_label_auc(y_true, y_pred)
    return float(np.nanmean(aucs))
