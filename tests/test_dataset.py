from pathlib import Path

import pandas as pd
import pytest
import torch

from knee.dataset import KneeStudyDataset, StudyDecodeError
from knee.infer import LABEL_COLUMNS, build_submission

_SAMPLE_ROOT = Path(__file__).resolve().parents[1] / "data" / "sample"
_TEST_SERIES_DIR = _SAMPLE_ROOT / "test_series"
_TEST_SERIES_CSV = _SAMPLE_ROOT / "test_series.csv"

_HAS_REAL_STUDY = _TEST_SERIES_DIR.exists() and any(_TEST_SERIES_DIR.iterdir())


def _real_study_uid():
    return next(p.name for p in _TEST_SERIES_DIR.iterdir() if p.is_dir())


@pytest.mark.skipif(not _HAS_REAL_STUDY, reason="no real downloaded study available")
def test_returns_correct_image_tensor_shape_and_dtype():
    series_df = pd.read_csv(_TEST_SERIES_CSV)
    study_uid = _real_study_uid()

    dataset = KneeStudyDataset(
        study_uids=[study_uid],
        dcm_root=_TEST_SERIES_DIR,
        series_df=series_df,
        n_slices=16,
        size=224,
    )
    image, labels, returned_uid = dataset[0]

    assert image.shape == (16, 3, 224, 224)
    assert image.dtype == torch.float32
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert returned_uid == study_uid
    assert labels is None


@pytest.mark.skipif(not _HAS_REAL_STUDY, reason="no real downloaded study available")
def test_pads_when_series_has_fewer_slices_than_requested():
    series_df = pd.read_csv(_TEST_SERIES_CSV)
    study_uid = _real_study_uid()

    # the smallest real series here has 30 files; ask for more than any series has
    dataset = KneeStudyDataset(
        study_uids=[study_uid],
        dcm_root=_TEST_SERIES_DIR,
        series_df=series_df,
        n_slices=200,
        size=224,
    )
    image, _, _ = dataset[0]

    assert image.shape == (200, 3, 224, 224)


@pytest.mark.skipif(not _HAS_REAL_STUDY, reason="no real downloaded study available")
def test_returns_nan_labels_when_study_missing_from_labels_df():
    series_df = pd.read_csv(_TEST_SERIES_CSV)
    study_uid = _real_study_uid()
    labels_df = pd.DataFrame(
        [["some_other_study"] + [1.0] * len(LABEL_COLUMNS)],
        columns=["StudyInstanceUID"] + LABEL_COLUMNS,
    )

    dataset = KneeStudyDataset(
        study_uids=[study_uid],
        dcm_root=_TEST_SERIES_DIR,
        series_df=series_df,
        labels_df=labels_df,
        n_slices=16,
        size=224,
    )
    _, labels, _ = dataset[0]

    assert torch.isnan(labels).all()


def test_raises_study_decode_error_when_study_has_no_series_rows():
    # study_uid has zero rows in series_df -- select_series returns [], and
    # indexing [0] into that used to raise a bare IndexError
    series_df = pd.DataFrame(
        columns=["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"]
    )

    dataset = KneeStudyDataset(
        study_uids=["study_with_no_series_metadata"],
        dcm_root=Path("/nonexistent"),
        series_df=series_df,
        n_slices=16,
        size=224,
    )

    with pytest.raises(StudyDecodeError):
        dataset[0]


@pytest.mark.skipif(not _HAS_REAL_STUDY, reason="no real downloaded study available")
def test_raises_study_decode_error_when_series_directory_has_no_dcm_files(tmp_path):
    series_df = pd.read_csv(_TEST_SERIES_CSV)
    study_uid = _real_study_uid()
    real_series_uid = series_df[series_df["StudyInstanceUID"] == study_uid]["SeriesInstanceUID"].iloc[0]

    # an empty series directory (e.g. a partially-completed download) used to
    # crash on slices[-1] in the padding loop with an IndexError
    empty_series_dir = tmp_path / study_uid / real_series_uid
    empty_series_dir.mkdir(parents=True)

    dataset = KneeStudyDataset(
        study_uids=[study_uid],
        dcm_root=tmp_path,
        series_df=series_df,
        n_slices=16,
        size=224,
    )

    with pytest.raises(StudyDecodeError):
        dataset[0]


def test_build_submission_still_falls_back_to_0_5_when_dataset_raises_study_decode_error():
    series_df = pd.DataFrame(
        columns=["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"]
    )
    dataset = KneeStudyDataset(
        study_uids=["broken_study"],
        dcm_root=Path("/nonexistent"),
        series_df=series_df,
        n_slices=16,
        size=224,
    )

    def predict_fn(study_uid):
        image, _, _ = dataset[0]
        return image  # never reached for the broken study

    df = build_submission(["broken_study"], predict_fn)

    assert (df[LABEL_COLUMNS].to_numpy()[0] == 0.5).all()
