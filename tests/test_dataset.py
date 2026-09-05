from pathlib import Path

import pandas as pd
import pytest
import torch

import numpy as np

from knee.dataset import (
    CachedDataset,
    KneeStudyDataset,
    PreppedStudyDataset,
    StudyDecodeError,
    augment_volume,
)
from knee.infer import LABEL_COLUMNS, build_submission
from knee.prep import save_study_npz

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


def _write_prepped_study(npz_root, study_uid, n_series=2, n_slices=4, size=16, side="R",
                         planes=None):
    npz_root.mkdir(parents=True, exist_ok=True)
    series_slices = {
        f"series-{i}": [
            np.random.randint(0, 255, (size, size), dtype=np.uint8) for _ in range(n_slices)
        ]
        for i in range(n_series)
    }
    planes = planes or ["Coronal"] * n_series
    meta = {
        "StudyInstanceUID": study_uid,
        "side": side,
        "route": "Laterality",
        "is_gold": False,
        "series": {
            f"series-{i}": {"Anatomical_Plane": planes[i]} for i in range(n_series)
        },
    }
    save_study_npz(npz_root / f"{study_uid}.npz", series_slices, meta)
    return series_slices


def test_prepped_dataset_returns_the_same_tensor_contract_as_the_raw_dataset(tmp_path):
    _write_prepped_study(tmp_path, "study-a", n_series=2, n_slices=4, size=16)

    dataset = PreppedStudyDataset(["study-a"], tmp_path, n_slices=4, max_series=4)
    image, labels, returned_uid = dataset[0]

    assert image.shape == (16, 3, 16, 16)  # max_series * n_slices, padded on both axes
    assert image.dtype == torch.float32
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert returned_uid == "study-a"
    assert labels is None
    # the three channels are an expanded view, so they must still read as three
    # identical copies of the grayscale slice
    assert torch.equal(image[:, 0], image[:, 2])


def test_prepped_dataset_pads_short_series_by_repeating_the_last_slice(tmp_path):
    # artifacts store the true slice count (census p0 = 11, below K=24), so
    # padding has to happen here rather than being baked into the .npz
    _write_prepped_study(tmp_path, "short", n_series=1, n_slices=3, size=8)

    dataset = PreppedStudyDataset(["short"], tmp_path, n_slices=6, max_series=1)
    image, _, _ = dataset[0]

    assert image.shape == (6, 3, 8, 8)
    assert torch.equal(image[3], image[5])  # the repeated tail


def test_prepped_dataset_mirrors_left_studies_and_leaves_right_ones_alone(tmp_path):
    # np.fliplr returns a negative-stride view that torch.from_numpy rejects,
    # so this also checks the flip survives the stack into a tensor at all
    stored = _write_prepped_study(tmp_path, "left", n_series=1, n_slices=1, size=8, side="L")
    _write_prepped_study(tmp_path, "right", n_series=1, n_slices=1, size=8, side="R")

    left_image, _, _ = PreppedStudyDataset(["left"], tmp_path, n_slices=1, max_series=1)[0]
    right_image, _, _ = PreppedStudyDataset(["right"], tmp_path, n_slices=1, max_series=1)[0]

    original = stored["series-0"][0]
    flipped = torch.from_numpy(np.fliplr(original).copy().astype(np.float32) / 255.0)
    # JPEG q=92 is lossy, so compare within a tolerance rather than exactly
    assert (left_image[0, 0] - flipped).abs().mean() < 8 / 255.0
    assert not torch.equal(left_image, right_image)


def test_prepped_dataset_does_not_flip_sagittal_series(tmp_path):
    # A sagittal image's horizontal axis is anterior-posterior (row direction
    # runs along patient +y), not medial-lateral -- flipping one mirrors the
    # knee front-to-back rather than swapping sides. Only Axial/Coronal (row
    # direction along +x) can be laterality-canonicalized in plane. Caught by
    # the Phase 2 visual check; artifacts store unmirrored pixels, so the fix
    # was loader-only.
    stored = _write_prepped_study(
        tmp_path, "left", n_series=2, n_slices=1, size=8, side="L",
        planes=["Sagittal", "Coronal"],
    )

    image, _, _ = PreppedStudyDataset(["left"], tmp_path, n_slices=1, max_series=2)[0]

    tol = 8 / 255.0  # JPEG q=92 is lossy
    sagittal = torch.from_numpy(stored["series-0"][0].astype(np.float32) / 255.0)
    coronal = torch.from_numpy(np.fliplr(stored["series-1"][0]).copy().astype(np.float32) / 255.0)

    assert (image[0, 0] - sagittal).abs().mean() < tol  # sagittal: unflipped
    assert (image[1, 0] - coronal).abs().mean() < tol  # coronal: flipped


def test_prepped_dataset_raises_study_decode_error_when_the_artifact_is_missing(tmp_path):
    dataset = PreppedStudyDataset(["never-prepped"], tmp_path)

    with pytest.raises(StudyDecodeError):
        dataset[0]


def test_prepped_dataset_returns_nan_labels_when_study_missing_from_labels_df(tmp_path):
    _write_prepped_study(tmp_path, "study-a", n_series=1, n_slices=2, size=8)
    labels_df = pd.DataFrame(
        [["some_other_study"] + [1.0] * len(LABEL_COLUMNS)],
        columns=["StudyInstanceUID"] + LABEL_COLUMNS,
    )

    dataset = PreppedStudyDataset(["study-a"], tmp_path, labels_df=labels_df, n_slices=2)
    _, labels, _ = dataset[0]

    assert torch.isnan(labels).all()


class _CountingBaseDataset:
    """Fake base dataset that counts how many times __getitem__ actually ran,
    so CachedDataset's "decode at most once" guarantee can be verified directly."""

    def __init__(self, n_items=3):
        self.n_items = n_items
        self.call_count = 0

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        self.call_count += 1
        return f"item{idx}", idx


def test_cached_dataset_returns_the_same_item_as_the_base_dataset():
    base = _CountingBaseDataset()
    cached = CachedDataset(base)

    assert cached[1] == ("item1", 1)
    assert len(cached) == len(base)


def test_cached_dataset_only_computes_each_item_once():
    base = _CountingBaseDataset()
    cached = CachedDataset(base)

    cached[0]
    cached[0]
    cached[0]

    assert base.call_count == 1


def test_prepped_dataset_side_override_mirrors_a_study_the_artifact_left_unresolved(tmp_path):
    """Phase 2 prepped 49% of the corpus with side=None because no tag resolved.
    The geometry route recovers those sides (NOTES 2026-09-05), and the pixels
    were stored unmirrored -- so the correction lands here, at load time, rather
    than costing a re-prep of all 4,407 studies."""
    stored = _write_prepped_study(tmp_path, "unresolved", n_series=1, n_slices=1,
                                  size=8, side=None, planes=["Coronal"])

    without = PreppedStudyDataset(["unresolved"], tmp_path, n_slices=1, max_series=1)[0][0]
    with_side = PreppedStudyDataset(["unresolved"], tmp_path, n_slices=1, max_series=1,
                                    sides={"unresolved": "L"})[0][0]

    original = stored["series-0"][0]
    flipped = torch.from_numpy(np.fliplr(original).copy().astype(np.float32) / 255.0)
    assert (with_side[0, 0] - flipped).abs().mean() < 8 / 255.0
    assert not torch.equal(without, with_side)


def test_prepped_dataset_side_override_wins_over_the_stored_side(tmp_path):
    _write_prepped_study(tmp_path, "s", n_series=1, n_slices=1, size=8, side="R",
                         planes=["Coronal"])

    stored_side = PreppedStudyDataset(["s"], tmp_path, n_slices=1, max_series=1)[0][0]
    overridden = PreppedStudyDataset(["s"], tmp_path, n_slices=1, max_series=1,
                                     sides={"s": "L"})[0][0]

    assert not torch.equal(stored_side, overridden)


def test_prepped_dataset_falls_back_to_the_stored_side_for_studies_not_in_the_map(tmp_path):
    """A partial map must not silently un-mirror every study missing from it."""
    _write_prepped_study(tmp_path, "s", n_series=1, n_slices=1, size=8, side="L",
                         planes=["Coronal"])

    stored_side = PreppedStudyDataset(["s"], tmp_path, n_slices=1, max_series=1)[0][0]
    partial = PreppedStudyDataset(["s"], tmp_path, n_slices=1, max_series=1,
                                  sides={"other": "R"})[0][0]

    assert torch.equal(stored_side, partial)


# --- augmentation (Phase 5 screen B) -----------------------------------------
# Screen A measured train stable-6 0.979 against val 0.836 at 8 epochs, so the
# binding constraint is overfitting and this is the lever for it.

def test_augment_volume_changes_the_pixels():
    torch.manual_seed(0)
    volume = torch.rand(4, 32, 32)

    out = augment_volume(volume)

    assert out.shape == volume.shape
    assert out.dtype == volume.dtype
    assert not torch.equal(out, volume)


def test_augment_volume_applies_one_transform_to_every_slice():
    """A study is one sample. Jittering slices independently would desynchronise
    the anatomy down the stack, which is noise rather than augmentation."""
    torch.manual_seed(0)
    slice_ = torch.rand(1, 32, 32)
    volume = slice_.repeat(5, 1, 1)          # five identical slices

    out = augment_volume(volume)

    for i in range(1, 5):
        assert torch.equal(out[0], out[i]), f"slice {i} got a different transform"


def test_augment_volume_keeps_values_in_range():
    torch.manual_seed(0)
    volume = torch.rand(4, 32, 32)

    for _ in range(10):
        out = augment_volume(volume)
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_augment_volume_never_mirrors():
    """Laterality is canonicalised upstream (every knee mapped to one side), so a
    horizontal flip here would undo exactly the signal four labels depend on."""
    torch.manual_seed(0)
    # a left-right asymmetric volume: bright on one side only
    volume = torch.zeros(2, 16, 16)
    volume[:, :, :4] = 1.0

    for _ in range(20):
        out = augment_volume(volume)
        assert out[:, :, :8].sum() > out[:, :, 8:].sum(), "augmentation mirrored the volume"


def test_prepped_dataset_augments_only_when_asked(tmp_path):
    _write_prepped_study(tmp_path, "s", n_series=1, n_slices=2, size=16)

    torch.manual_seed(0)
    plain = PreppedStudyDataset(["s"], tmp_path, n_slices=2, max_series=1)[0][0]
    torch.manual_seed(0)
    again = PreppedStudyDataset(["s"], tmp_path, n_slices=2, max_series=1)[0][0]
    torch.manual_seed(0)
    augmented = PreppedStudyDataset(["s"], tmp_path, n_slices=2, max_series=1, augment=True)[0][0]

    assert torch.equal(plain, again), "the un-augmented path must stay deterministic"
    assert not torch.equal(plain, augmented)
    assert augmented.shape == plain.shape
