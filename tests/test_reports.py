import numpy as np
import pandas as pd

from knee.infer import LABEL_COLUMNS
from knee.reports import build_lexical_labels, lexical_label


def test_positive_match_with_no_preceding_negation_is_labeled_1():
    text = "There is a large effusion in the suprapatellar recess."

    assert lexical_label(text, "effusion") == 1


def test_negation_immediately_before_match_is_labeled_0():
    text = "No effusion is seen."

    assert lexical_label(text, "effusion") == 0


def test_no_match_at_all_returns_none_not_negative():
    text = "Normal knee MRI, no acute findings."

    assert lexical_label(text, "effusion") is None


def test_spanish_positive_match():
    text = "Derrame articular moderado en el compartimento medial."

    assert lexical_label(text, "effusion") == 1


def test_spanish_negation():
    text = "No hay derrame articular."

    assert lexical_label(text, "effusion") == 0


def test_build_lexical_labels_produces_all_12_columns_nan_for_unsupported():
    df = pd.DataFrame(
        {
            "StudyInstanceUID": ["s1"],
            "Report": ["There is a large effusion. No acute fracture."],
        }
    )

    labels = build_lexical_labels(df)

    assert list(labels.columns) == ["StudyInstanceUID"] + LABEL_COLUMNS
    assert labels.loc[0, "Effusion"] == 1
    # Synovitis has no lexical rule defined at all -- must be NaN, not crash
    assert np.isnan(labels.loc[0, "Synovitis"])


def test_build_lexical_labels_is_nan_for_studies_with_no_rule_match():
    df = pd.DataFrame(
        {
            "StudyInstanceUID": ["s1"],
            "Report": ["Normal knee MRI, no acute findings."],
        }
    )

    labels = build_lexical_labels(df)

    assert np.isnan(labels.loc[0, "Effusion"])


def test_build_lexical_labels_handles_nan_report_without_crashing():
    df = pd.DataFrame(
        {
            "StudyInstanceUID": ["s1", "s2"],
            "Report": [np.nan, "There is a large effusion."],
        }
    )

    labels = build_lexical_labels(df)

    assert np.isnan(labels.loc[0, "Effusion"])
    assert labels.loc[1, "Effusion"] == 1
