from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knee.infer import LABEL_COLUMNS, build_submission, rank_mean

_SAMPLE_SUBMISSION = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_submission.csv"


def test_label_columns_match_required_header_byte_for_byte():
    expected_header = _SAMPLE_SUBMISSION.read_text().splitlines()[0]
    actual_header = ",".join(["StudyInstanceUID"] + LABEL_COLUMNS)

    assert actual_header == expected_header


def test_build_submission_one_row_per_study_values_in_unit_interval():
    study_uids = ["study1", "study2", "study3"]

    def predict_fn(uid):
        return np.full(len(LABEL_COLUMNS), 0.7)

    df = build_submission(study_uids, predict_fn)

    assert len(df) == 3
    assert list(df["StudyInstanceUID"]) == study_uids
    values = df[LABEL_COLUMNS].to_numpy()
    assert not np.isnan(values).any()
    assert (values >= 0).all() and (values <= 1).all()


def test_build_submission_falls_back_to_0_5_on_prediction_failure():
    study_uids = ["good_study", "broken_study"]

    def predict_fn(uid):
        if uid == "broken_study":
            raise RuntimeError("DICOM decode failed")
        return np.full(len(LABEL_COLUMNS), 0.9)

    df = build_submission(study_uids, predict_fn)

    broken_row = df[df["StudyInstanceUID"] == "broken_study"][LABEL_COLUMNS].to_numpy()[0]
    assert (broken_row == 0.5).all()


def test_written_csv_header_matches_sample_submission_byte_for_byte(tmp_path):
    study_uids = ["study1"]

    def predict_fn(uid):
        return np.full(len(LABEL_COLUMNS), 0.5)

    df = build_submission(study_uids, predict_fn)
    out_path = tmp_path / "submission.csv"
    df.to_csv(out_path, index=False)

    expected_header = _SAMPLE_SUBMISSION.read_text().splitlines()[0]
    actual_header = out_path.read_text().splitlines()[0]
    assert actual_header == expected_header


def test_rank_mean_preserves_a_single_model_ordering_and_stays_in_range():
    # AUC reads order only, so a rank transform of one model must not change
    # its score -- but it does change the numbers, and a submission still has
    # to be in [0, 1].
    preds = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])

    out = rank_mean([preds])

    assert out.shape == preds.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    for col in range(preds.shape[1]):
        assert list(np.argsort(out[:, col])) == list(np.argsort(preds[:, col]))


def test_rank_mean_is_not_captured_by_the_most_confident_member():
    # The reason to prefer it over a probability mean: one overconfident member
    # can outvote two others in probability space while carrying one vote in
    # rank space. Here A is confident and wrong about the last two studies,
    # B and C are diffident and right.
    a = np.array([[0.01], [0.02], [0.99], [0.60]])
    b = np.array([[0.40], [0.45], [0.50], [0.55]])
    c = np.array([[0.41], [0.46], [0.51], [0.56]])

    prob_mean = np.mean([a, b, c], axis=0)[:, 0]
    ranked = rank_mean([a, b, c])[:, 0]

    assert prob_mean[2] > prob_mean[3], "probability mean should inherit A's error"
    assert ranked[3] > ranked[2], "rank mean should follow the B/C majority"


def test_rank_mean_gives_tied_predictions_equal_ranks():
    # ordinal ranking would break these arbitrarily and invent an ordering the
    # models never expressed
    preds = np.array([[0.5], [0.5], [0.9]])

    out = rank_mean([preds])[:, 0]

    assert out[0] == out[1] < out[2]


def test_rank_mean_rejects_members_of_different_shapes():
    with pytest.raises(ValueError):
        rank_mean([np.zeros((3, 2)), np.zeros((4, 2))])
