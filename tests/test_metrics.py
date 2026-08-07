import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from knee.metrics import macro_auc, per_label_auc


def test_macro_auc_matches_sklearn_macro_average():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=(200, 12))
    y_pred = rng.rand(200, 12)

    expected = roc_auc_score(y_true, y_pred, average="macro")
    actual = macro_auc(y_true, y_pred)

    assert actual == pytest.approx(expected)


def test_per_label_auc_matches_sklearn_per_column():
    rng = np.random.RandomState(1)
    y_true = rng.randint(0, 2, size=(200, 3))
    y_pred = rng.rand(200, 3)

    expected = [roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(3)]
    actual = per_label_auc(y_true, y_pred)

    assert list(actual) == pytest.approx(expected)


def test_handles_label_with_single_class_present():
    y_true = np.array([[0, 1], [0, 0], [0, 1], [0, 0]])
    y_pred = np.array([[0.1, 0.9], [0.2, 0.1], [0.3, 0.8], [0.4, 0.2]])

    # column 0 is all zeros -> undefined AUC, must not raise
    labels = per_label_auc(y_true, y_pred)

    assert np.isnan(labels[0])
    assert not np.isnan(labels[1])


def test_macro_auc_excludes_undefined_labels_from_the_average():
    y_true = np.array([[0, 1], [0, 0], [0, 1], [0, 0]])
    y_pred = np.array([[0.1, 0.9], [0.2, 0.1], [0.3, 0.8], [0.4, 0.2]])

    expected = roc_auc_score(y_true[:, 1], y_pred[:, 1])
    actual = macro_auc(y_true, y_pred)

    assert actual == pytest.approx(expected)


def test_handles_nan_targets_from_unmatched_lexical_labels():
    # lexical_label returns None for studies with no rule match, which becomes
    # NaN once collected into a float array -- must not raise, and must score
    # only the rows that actually have a target for that label.
    nan = np.nan
    y_true = np.array([[1.0], [0.0], [nan], [1.0], [nan], [0.0]])
    y_pred = np.array([[0.9], [0.2], [0.5], [0.8], [0.5], [0.4]])

    labeled_mask = ~np.isnan(y_true[:, 0])
    expected = roc_auc_score(y_true[labeled_mask, 0], y_pred[labeled_mask, 0])

    labels = per_label_auc(y_true, y_pred)

    assert labels[0] == pytest.approx(expected)


def test_macro_auc_handles_all_nan_column():
    nan = np.nan
    y_true = np.array([[1.0, nan], [0.0, nan], [1.0, nan]])
    y_pred = np.array([[0.9, 0.5], [0.2, 0.5], [0.8, 0.5]])

    expected = roc_auc_score(y_true[:, 0], y_pred[:, 0])
    actual = macro_auc(y_true, y_pred)

    assert actual == pytest.approx(expected)
