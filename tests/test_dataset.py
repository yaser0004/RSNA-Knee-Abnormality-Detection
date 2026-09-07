from pathlib import Path

import pandas as pd
import pytest
import torch

import numpy as np

from knee.dataset import (
    CachedDataset,
    KneeStudyDataset,
    PreppedSlotDataset,
    PreppedStudyDataset,
    StudyDecodeError,
    augment_volume,
)
from knee.dicom import SLICE_GROUP, SLOTS
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


# --- Phase 6: slot-structured loading with a presence mask ----------------------


def _slot_artifact(tmp_path, uid="study", filled=("SAG_FLUID", "COR_FLUID"), n_groups=2,
                   size=8, side="L"):
    """Write a v2 artifact directly, so these tests exercise the loader rather
    than the DICOM path."""
    slot_slices, series_meta = {}, {}
    for i, name in enumerate(filled):
        slot_slices[name] = [
            np.full((size, size), (i * 10 + j) % 256, dtype=np.uint8)
            for j in range(n_groups * SLICE_GROUP)
        ]
        series_meta[f"series-{i}"] = {"slot": name, "n_slices_stored": len(slot_slices[name])}
    meta = {
        "StudyInstanceUID": uid, "side": side, "route": "Laterality", "is_gold": False,
        "slots": {name: (f"series-{filled.index(name)}" if name in filled else None)
                  for name, _, _ in SLOTS},
        "series": series_meta,
    }
    save_study_npz(tmp_path / f"{uid}.npz", slot_slices, meta)
    return uid


def test_slot_dataset_returns_a_real_slot_axis_and_a_presence_mask(tmp_path):
    uid = _slot_artifact(tmp_path, filled=("SAG_FLUID", "AX_FLUID"), n_groups=2, size=8)

    ds = PreppedSlotDataset([uid], tmp_path, n_groups=2, out_size=8)
    image, mask, labels, _ = ds[0]

    # (slot, group, channel, H, W) -- the flat volume v1 returned could not
    # express "this study has no coronal series" at all
    assert tuple(image.shape) == (len(SLOTS), 2, SLICE_GROUP, 8, 8)
    assert tuple(mask.shape) == (len(SLOTS),)
    assert mask.dtype == torch.bool


def test_slot_dataset_marks_an_absent_slot_false_and_zeroes_it(tmp_path):
    uid = _slot_artifact(tmp_path, filled=("SAG_FLUID",))

    image, mask, _, _ = PreppedSlotDataset([uid], tmp_path, n_groups=2, out_size=8)[0]

    names = [name for name, _, _ in SLOTS]
    assert bool(mask[names.index("SAG_FLUID")]) is True
    for absent in ("COR_FLUID", "AX_FLUID", "SAG_STRUCT", "COR_STRUCT", "AX_STRUCT"):
        i = names.index(absent)
        assert bool(mask[i]) is False
        assert float(image[i].abs().sum()) == 0.0


def test_slot_dataset_puts_three_adjacent_stored_slices_in_the_channel_axis(tmp_path):
    # the point of storing contiguous groups: the channels carry local
    # through-plane context, not one slice replicated three times
    uid = "adjacent"
    slot_slices = {"SAG_FLUID": [np.full((4, 4), v, dtype=np.uint8) for v in (10, 20, 30)]}
    meta = {"StudyInstanceUID": uid, "side": None, "route": "unknown", "is_gold": False,
            "slots": {name: ("s0" if name == "SAG_FLUID" else None) for name, _, _ in SLOTS},
            "series": {"s0": {"slot": "SAG_FLUID", "n_slices_stored": 3}}}
    save_study_npz(tmp_path / f"{uid}.npz", slot_slices, meta)

    image, _, _, _ = PreppedSlotDataset([uid], tmp_path, n_groups=1, out_size=4)[0]

    channels = [float(image[0, 0, c].mean() * 255) for c in range(SLICE_GROUP)]
    assert channels == pytest.approx([10, 20, 30], abs=1.0)
    assert len(set(channels)) == 3, "channels must differ -- not one slice tripled"


def test_slot_dataset_orders_slots_by_the_table_not_by_storage_order(tmp_path):
    uid = _slot_artifact(tmp_path, filled=("AX_STRUCT", "SAG_FLUID"))

    _, mask, _, _ = PreppedSlotDataset([uid], tmp_path, n_groups=2, out_size=8)[0]

    names = [name for name, _, _ in SLOTS]
    assert bool(mask[names.index("SAG_FLUID")]) and bool(mask[names.index("AX_STRUCT")])
    assert not bool(mask[names.index("COR_FLUID")])


def test_slot_dataset_mirrors_only_the_planes_where_left_right_is_in_plane(tmp_path):
    # sagittal's horizontal axis is anterior-posterior, so flipping it mirrors
    # the knee front-to-back -- the bug mirrors_in_plane exists to prevent
    uid = "sided"
    asym = np.tile(np.arange(8, dtype=np.uint8) * 30, (8, 1))
    slot_slices = {name: [asym.copy() for _ in range(SLICE_GROUP)]
                   for name in ("SAG_FLUID", "COR_FLUID")}
    meta = {"StudyInstanceUID": uid, "side": "L", "route": "Laterality", "is_gold": False,
            "slots": {name: (name if name in slot_slices else None) for name, _, _ in SLOTS},
            "series": {name: {"slot": name, "n_slices_stored": SLICE_GROUP}
                       for name in slot_slices}}
    save_study_npz(tmp_path / f"{uid}.npz", slot_slices, meta)

    image, _, _, _ = PreppedSlotDataset([uid], tmp_path, n_groups=1, out_size=8)[0]

    names = [name for name, _, _ in SLOTS]
    sag = image[names.index("SAG_FLUID"), 0, 0].numpy()
    cor = image[names.index("COR_FLUID"), 0, 0].numpy()
    # atol covers the artifact's JPEG q=92 round trip, which perturbs a smooth
    # gradient by a few thousandths; a flip would move it by whole steps
    assert np.allclose(sag, asym / 255.0, atol=0.02), "sagittal must not be flipped"
    assert np.allclose(cor, np.fliplr(asym) / 255.0, atol=0.02), "coronal must be flipped"
    assert sag[0, 0] < sag[0, -1] and cor[0, 0] > cor[0, -1]


def test_slot_dataset_downsamples_to_the_requested_size(tmp_path):
    # artifacts are stored at 336 so resolution stays screenable without a
    # second prep run
    uid = _slot_artifact(tmp_path, size=16)

    image, _, _, _ = PreppedSlotDataset([uid], tmp_path, n_groups=2, out_size=8)[0]

    assert image.shape[-2:] == (8, 8)


def test_slot_dataset_spreads_fewer_groups_across_the_stored_band(tmp_path):
    # Artifacts store 5 groups across the 0.20-0.80 band. A cell asking for 3
    # must get groups spread over that band, not its first three -- otherwise
    # "fewer groups" is confounded with "only the bottom of the knee", and a
    # screen varying n_groups would measure the wrong thing.
    uid = "spread"
    # 5 groups of 3, each group uniformly valued by its index
    slot_slices = {"SAG_FLUID": [np.full((4, 4), g * 50, dtype=np.uint8)
                                 for g in range(5) for _ in range(SLICE_GROUP)]}
    meta = {"StudyInstanceUID": uid, "side": None, "route": "unknown", "is_gold": False,
            "slots": {n: ("s0" if n == "SAG_FLUID" else None) for n, _, _ in SLOTS},
            "series": {"s0": {"slot": "SAG_FLUID", "n_slices_stored": 15}}}
    save_study_npz(tmp_path / f"{uid}.npz", slot_slices, meta)

    image, _, _, _ = PreppedSlotDataset([uid], tmp_path, n_groups=3, out_size=4,
                                        slots=["SAG_FLUID"])[0]

    groups = [round(float(image[0, g, 0].mean() * 255) / 50) for g in range(3)]
    assert groups == [0, 2, 4], f"expected the band's ends and middle, got {groups}"


class TestSlotDatasetSampleWeights:
    """PreppedSlotDataset optionally yields a per-label sample weight alongside
    the target, so the confidence screen can discount rows the label sources
    disagree about. Without a weights_df the batch shape is unchanged -- every
    config screened so far pairs against runs that produced the 4-tuple."""

    def test_without_weights_the_batch_shape_is_unchanged(self, tmp_path):
        uid = _slot_artifact(tmp_path, filled=("SAG_FLUID",))
        assert len(PreppedSlotDataset([uid], tmp_path, n_groups=2, out_size=8)[0]) == 4

    def test_with_weights_the_weight_row_rides_alongside_the_target(self, tmp_path):
        uid = _slot_artifact(tmp_path, filled=("SAG_FLUID",))
        labels = pd.DataFrame([[uid] + [1.0] * len(LABEL_COLUMNS)],
                              columns=["StudyInstanceUID"] + LABEL_COLUMNS)
        weights = pd.DataFrame([[uid] + [0.5] * len(LABEL_COLUMNS)],
                               columns=["StudyInstanceUID"] + LABEL_COLUMNS)

        image, mask, y, w, out_uid = PreppedSlotDataset(
            [uid], tmp_path, labels_df=labels, weights_df=weights,
            n_groups=2, out_size=8)[0]

        assert out_uid == uid
        assert w.shape == y.shape
        assert torch.allclose(w, torch.full_like(w, 0.5))

    def test_a_study_absent_from_the_weights_gets_full_weight_not_zero(self, tmp_path):
        """A missing weight means "no agreement information", which is not the
        same as "this study is worthless" -- zero would silently drop it."""
        uid = _slot_artifact(tmp_path, filled=("SAG_FLUID",))
        labels = pd.DataFrame([[uid] + [1.0] * len(LABEL_COLUMNS)],
                              columns=["StudyInstanceUID"] + LABEL_COLUMNS)

        _, _, y, w, _ = PreppedSlotDataset(
            [uid], tmp_path, labels_df=labels,
            weights_df=pd.DataFrame(columns=["StudyInstanceUID"] + LABEL_COLUMNS),
            n_groups=2, out_size=8)[0]

        assert torch.allclose(w, torch.ones_like(w))
