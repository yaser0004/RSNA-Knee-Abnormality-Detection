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


def paired_macro_auc_delta(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Macro-AUC delta (b - a) with a bootstrap 95% CI, resampling studies.

    The promotion rule the plan binds every experiment to is "the paired
    per-study delta clears noise", and a single macro-AUC difference cannot say
    whether it does. Both arms are scored on the *same* resampled studies in
    each replicate -- that pairing is the whole point, since the two models
    share their hard cases and an unpaired CI would be far too wide to resolve
    the few-thousandths deltas Phase 5 is chasing.

    Returns (delta, ci_low, ci_high) over the non-NaN entries of each column."""
    rng = np.random.default_rng(seed)
    delta = macro_auc(y_true, y_pred_b) - macro_auc(y_true, y_pred_a)

    n = len(y_true)
    replicates = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        replicates[i] = macro_auc(yt, y_pred_b[idx]) - macro_auc(yt, y_pred_a[idx])
    lo, hi = np.percentile(replicates, [2.5, 97.5])
    return float(delta), float(lo), float(hi)
