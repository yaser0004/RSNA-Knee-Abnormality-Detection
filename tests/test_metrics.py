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
