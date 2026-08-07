from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knee.infer import LABEL_COLUMNS, build_submission

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
