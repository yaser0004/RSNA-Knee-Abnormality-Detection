import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knee.dicom import StudyDecodeError
from knee.prep import (
    _pad_to_square,
    load_study_npz,
    mirror_to_canonical,
    mirrors_in_plane,
    normalize_series,
    prep_study,
    save_study_npz,
)

_SAMPLE_ROOT = Path(__file__).resolve().parents[1] / "data" / "sample"
_TEST_SERIES_DIR = _SAMPLE_ROOT / "test_series"
_TEST_SERIES_CSV = _SAMPLE_ROOT / "test_series.csv"

_HAS_REAL_STUDY = _TEST_SERIES_DIR.exists() and any(_TEST_SERIES_DIR.iterdir())


def _real_study_uid():
    return next(p.name for p in sorted(_TEST_SERIES_DIR.iterdir()) if p.is_dir())


def test_normalize_series_clips_using_stats_from_the_whole_series_not_one_slice():
    # slice 0 is uniformly bright, slice 1 has the real dynamic range -- a
    # per-slice clip would flatten slice 0 to a constant; a per-series clip
    # (this function) should place slice 0 near the bright end instead, since
    # its values sit inside the series-wide percentile range.
    bright_slice = np.full((4, 4), 900.0, dtype=np.float32)
    varied_slice = np.linspace(0, 1000, 16, dtype=np.float32).reshape(4, 4)

    normalized = normalize_series([bright_slice, varied_slice])

    assert len(normalized) == 2
    for arr in normalized:
        assert arr.dtype == np.uint8
    assert normalized[0].std() < 5  # still near-constant (uniformly bright)
    assert normalized[0].mean() > 150  # but placed near the bright end, not flattened to 0


def test_normalize_series_handles_empty_list():
    assert normalize_series([]) == []


def test_normalize_series_handles_constant_series_without_dividing_by_zero():
    slices = [np.full((2, 2), 500.0, dtype=np.float32) for _ in range(3)]

    normalized = normalize_series(slices)

    for arr in normalized:
        assert arr.dtype == np.uint8
        assert (arr == 0).all()


def test_mirror_flips_left_studies_to_canonical_right():
    image = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    mirrored = mirror_to_canonical(image, side="L", canonical="R")

    assert np.array_equal(mirrored, np.fliplr(image))


def test_mirror_leaves_canonical_side_unchanged():
    image = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    result = mirror_to_canonical(image, side="R", canonical="R")

    assert np.array_equal(result, image)


def test_mirror_leaves_unresolved_laterality_unchanged():
    # side=None means resolve_laterality/resolve_study_laterality couldn't
    # determine handedness -- mirroring would be a guess, so this must be a
    # no-op; callers are responsible for excluding unresolved studies from
    # laterality-dependent training rather than mirroring blindly here.
    image = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    result = mirror_to_canonical(image, side=None, canonical="R")

    assert np.array_equal(result, image)


def test_mirror_leaves_bilateral_side_unchanged():
    # "B" (bilateral -- both knees in one series, seen once in the Phase 2 census)
    # isn't a mirrorable L/R side; flipping it would be a guess, same as side=None.
    image = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    result = mirror_to_canonical(image, side="B", canonical="R")

    assert np.array_equal(result, image)


def test_study_npz_round_trip_preserves_shape_dtype_and_metadata(tmp_path):
    series_slices = {
        "series-a": [np.random.randint(0, 255, (8, 8), dtype=np.uint8) for _ in range(3)],
        "series-b": [np.random.randint(0, 255, (8, 8), dtype=np.uint8) for _ in range(2)],
    }
    meta = {"side": "R", "route": "Laterality", "is_gold": False}
    path = tmp_path / "study.npz"

    save_study_npz(path, series_slices, meta)
    loaded_slices, loaded_meta = load_study_npz(path)

    assert loaded_meta == meta
    assert set(loaded_slices.keys()) == set(series_slices.keys())
    for series_uid, original in series_slices.items():
        reloaded = loaded_slices[series_uid]
        assert len(reloaded) == len(original)
        for orig_slice, reloaded_slice in zip(original, reloaded):
            assert reloaded_slice.shape == orig_slice.shape
            assert reloaded_slice.dtype == np.uint8
            # JPEG (q=92) is lossy -- bound the round-trip error rather than
            # requiring an exact match.
            assert np.abs(orig_slice.astype(int) - reloaded_slice.astype(int)).mean() < 8


def test_study_npz_round_trip_on_a_single_slice_series(tmp_path):
    series_slices = {"only-series": [np.zeros((4, 4), dtype=np.uint8)]}
    meta = {"side": None, "route": "unknown", "is_gold": True}
    path = tmp_path / "study.npz"

    save_study_npz(path, series_slices, meta)
    loaded_slices, loaded_meta = load_study_npz(path)

    assert loaded_meta == meta
    assert loaded_slices["only-series"][0].shape == (4, 4)


def test_only_axial_and_coronal_can_be_laterality_mirrored_in_plane():
    # verified against real ImageOrientationPatient cosines: Axial/Coronal rows
    # run along patient +x (medial-lateral, so fliplr swaps sides), Sagittal rows
    # run along +y (anterior-posterior, so fliplr mirrors front-to-back instead)
    assert mirrors_in_plane("Axial")
    assert mirrors_in_plane("Coronal")
    assert not mirrors_in_plane("Sagittal")
    assert not mirrors_in_plane(None)
    assert not mirrors_in_plane("Oblique")


def test_pad_to_square_centers_a_wide_slice_without_scaling_it():
    # 7.9% of real series are non-square, worst ratio 2:1 (640x1280) -- resizing
    # those straight to a square squashes the anatomy, unevenly across the
    # corpus. Padding first keeps the true aspect ratio through the resize.
    wide = np.full((2, 4), 200, dtype=np.uint8)

    padded = _pad_to_square(wide)

    assert padded.shape == (4, 4)
    assert padded.dtype == np.uint8
    assert (padded[1:3] == 200).all()  # original rows, centered
    assert (padded[0] == 0).all() and (padded[3] == 0).all()  # zero letterbox


def test_pad_to_square_centers_a_tall_slice():
    tall = np.full((4, 2), 200, dtype=np.uint8)

    padded = _pad_to_square(tall)

    assert padded.shape == (4, 4)
    assert (padded[:, 1:3] == 200).all()
    assert (padded[:, 0] == 0).all() and (padded[:, 3] == 0).all()


def test_pad_to_square_leaves_an_already_square_slice_untouched():
    square = np.arange(9, dtype=np.uint8).reshape(3, 3)

    assert _pad_to_square(square) is square


def test_pad_to_square_handles_an_odd_size_difference():
    # 320x300 is a real shape in the corpus -- an odd gap can't split evenly,
    # so this pins down that it doesn't crash or lose a row
    wide = np.full((3, 4), 7, dtype=np.uint8)

    padded = _pad_to_square(wide)

    assert padded.shape == (4, 4)
    assert (padded == 7).sum() == 12


@pytest.mark.skipif(not _HAS_REAL_STUDY, reason="no real downloaded study available")
def test_prep_study_pads_non_square_series_so_anatomy_is_not_squashed():
    series_df = pd.read_csv(_TEST_SERIES_CSV)
    study_uid = _real_study_uid()

    series_slices, meta = prep_study(
        study_uid, _TEST_SERIES_DIR, series_df, k_slices=2, size=64, max_series=4
    )

    non_square = [
        uid for uid, sm in meta["series"].items() if sm["Rows"] != sm["Columns"]
    ]
    assert non_square, "sample study should contain a non-square series (640x1280)"
    for uid in non_square:
        # stored square regardless, but via padding -- the letterboxed edge is
        # dark, which a squashed resize of real anatomy would not be
        for arr in series_slices[uid]:
            assert arr.shape == (64, 64)
        sm = meta["series"][uid]
        short_axis_is_rows = sm["Rows"] < sm["Columns"]
        edge = series_slices[uid][0][0] if short_axis_is_rows else series_slices[uid][0][:, 0]
        assert edge.max() == 0


def test_load_study_npz_decodes_only_the_requested_series_and_slices(tmp_path):
    # the point of the limits is skipping the JPEG decode, which dominates load
    # time -- so this checks what comes back, series by series, not just a count
    series_slices = {
        f"series-{i}": [np.full((4, 4), 10 * i + j, dtype=np.uint8) for j in range(5)]
        for i in range(3)
    }
    path = tmp_path / "study.npz"
    save_study_npz(path, series_slices, {"side": None, "route": "unknown"})

    loaded, _ = load_study_npz(path, max_series=2, max_slices=3)

    # no per-series plane/fluid-sensitive meta here, so every series ties on
    # priority rank and falls back to alphabetical-by-UID order
    assert sorted(loaded) == ["series-0", "series-1"]
    assert all(len(s) == 3 for s in loaded.values())
    assert loaded["series-1"][0].mean() == pytest.approx(10, abs=8)


def test_load_study_npz_orders_series_by_priority_not_alphabetically(tmp_path):
    # prep_study selects series by _SERIES_PRIORITY (sagittal-fluid-sensitive
    # first), but once stored they live in a dict keyed by SeriesInstanceUID --
    # alphabetical order can invert that. A study whose top-priority series
    # happens to sort last alphabetically must still come back first at
    # max_series=1, or PreppedStudyDataset(max_series=1) would silently hand
    # back an arbitrary series instead of the one Phase 1 trained on.
    series_slices = {
        "zzz-sagittal-fluid": [np.full((4, 4), 1, dtype=np.uint8)],
        "aaa-axial-fluid": [np.full((4, 4), 2, dtype=np.uint8)],
    }
    meta = {
        "side": None,
        "route": "unknown",
        "series": {
            "zzz-sagittal-fluid": {"Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 1},
            "aaa-axial-fluid": {"Anatomical_Plane": "Axial", "Fluid_Sensitive": 1},
        },
    }
    path = tmp_path / "study.npz"
    save_study_npz(path, series_slices, meta)

    loaded, _ = load_study_npz(path, max_series=1)

    assert set(loaded) == {"zzz-sagittal-fluid"}


def test_load_study_npz_limits_larger_than_the_artifact_are_not_an_error(tmp_path):
    series_slices = {"only": [np.zeros((4, 4), dtype=np.uint8) for _ in range(2)]}
    path = tmp_path / "study.npz"
    save_study_npz(path, series_slices, {"side": None, "route": "unknown"})

    loaded, _ = load_study_npz(path, max_series=4, max_slices=24)

    assert len(loaded["only"]) == 2  # the true count; padding is the loader's job


@pytest.mark.skipif(not _HAS_REAL_STUDY, reason="no real downloaded study available")
def test_prep_study_returns_up_to_max_series_of_k_resized_uint8_slices():
    series_df = pd.read_csv(_TEST_SERIES_CSV)
    study_uid = _real_study_uid()

    series_slices, meta = prep_study(
        study_uid, _TEST_SERIES_DIR, series_df, k_slices=8, size=64, max_series=3
    )

    assert 1 <= len(series_slices) <= 3
    for slices in series_slices.values():
        assert len(slices) == 8
        for arr in slices:
            assert arr.shape == (64, 64)
            assert arr.dtype == np.uint8
    assert meta["StudyInstanceUID"] == study_uid
    assert set(meta["series"]) == set(series_slices)


@pytest.mark.skipif(not _HAS_REAL_STUDY, reason="no real downloaded study available")
def test_prep_study_stores_the_true_slice_count_rather_than_padding_to_k():
    # padding at prep time would bake duplicate pixels into the artifact
    # permanently; PreppedStudyDataset pads at load instead. Ask for more
    # slices than any real series has to force the short path.
    series_df = pd.read_csv(_TEST_SERIES_CSV)
    study_uid = _real_study_uid()

    series_slices, meta = prep_study(
        study_uid, _TEST_SERIES_DIR, series_df, k_slices=500, size=64, max_series=1
    )

    series_uid, slices = next(iter(series_slices.items()))
    assert len(slices) < 500
    assert meta["series"][series_uid]["n_slices_stored"] == len(slices)
    assert meta["series"][series_uid]["n_slices_original"] == len(slices)


@pytest.mark.skipif(not _HAS_REAL_STUDY, reason="no real downloaded study available")
def test_prep_study_records_the_laterality_route_and_gold_flag():
    series_df = pd.read_csv(_TEST_SERIES_CSV)
    study_uid = _real_study_uid()

    _, meta = prep_study(
        study_uid, _TEST_SERIES_DIR, series_df, k_slices=4, size=32, max_series=1, is_gold=True
    )

    assert meta["is_gold"] is True
    # never absent: an unresolved study still carries a named route, so a
    # downstream filter can tell "unknown" from "this field wasn't written"
    assert meta["route"] in {
        "ImageLaterality",
        "Laterality",
        "SeriesDescription",
        "BodyPartExamined",
        "conflict",
        "unknown",
    }
    assert meta["side"] in {"R", "L", "B", "U", None}


@pytest.mark.skipif(not _HAS_REAL_STUDY, reason="no real downloaded study available")
def test_prep_study_records_the_acquisition_tags_the_decode_census_counts():
    # gate 3 of the Phase 2 pilot counts transfer syntaxes, photometric
    # interpretations and matrix sizes across every prepped series -- it can
    # only do that if prep_study reports them for the slices it actually read.
    series_df = pd.read_csv(_TEST_SERIES_CSV)
    study_uid = _real_study_uid()

    _, meta = prep_study(
        study_uid, _TEST_SERIES_DIR, series_df, k_slices=4, size=32, max_series=2
    )

    for series_meta in meta["series"].values():
        assert series_meta["TransferSyntaxUID"]
        assert series_meta["PhotometricInterpretation"]
        assert series_meta["Rows"] and series_meta["Columns"]


def test_prep_study_raises_study_decode_error_when_study_has_no_series_rows():
    series_df = pd.DataFrame(
        columns=["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"]
    )

    with pytest.raises(StudyDecodeError):
        prep_study("study_with_no_series_metadata", Path("/nonexistent"), series_df)


def test_prep_study_raises_study_decode_error_when_the_study_directory_is_missing(tmp_path):
    series_df = pd.DataFrame(
        [["ghost-study", "ghost-series", "Sagittal", 1, 1]],
        columns=[
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Anatomical_Plane",
            "Fluid_Sensitive",
            "Fat_Suppression",
        ],
    )

    with pytest.raises(StudyDecodeError):
        prep_study("ghost-study", tmp_path, series_df)


@pytest.mark.skipif(not _HAS_REAL_STUDY, reason="no real downloaded study available")
def test_prep_study_skips_an_unusable_series_instead_of_dropping_the_whole_study(tmp_path):
    # at max_series=4 a single empty series directory must not cost the study
    # its other three -- and the reason has to be recorded, because a silently
    # missing series is exactly the failure the pilot's counters exist to catch.
    series_df = pd.read_csv(_TEST_SERIES_CSV)
    study_uid = _real_study_uid()
    real_series = sorted(p for p in (_TEST_SERIES_DIR / study_uid).iterdir() if p.is_dir())

    study_dir = tmp_path / study_uid
    shutil.copytree(real_series[0], study_dir / real_series[0].name)
    (study_dir / real_series[1].name).mkdir(parents=True)

    kept = {real_series[0].name, real_series[1].name}
    subset_df = series_df[series_df["SeriesInstanceUID"].isin(kept)]

    series_slices, meta = prep_study(
        study_uid, tmp_path, subset_df, k_slices=4, size=32, max_series=4
    )

    assert set(series_slices) == {real_series[0].name}
    assert real_series[1].name in meta["skipped_series"]


@pytest.mark.skipif(not _HAS_REAL_STUDY, reason="no real downloaded study available")
def test_prep_study_raises_study_decode_error_when_every_series_is_unusable(tmp_path):
    series_df = pd.read_csv(_TEST_SERIES_CSV)
    study_uid = _real_study_uid()
    real_series = sorted(p for p in (_TEST_SERIES_DIR / study_uid).iterdir() if p.is_dir())

    study_dir = tmp_path / study_uid
    for series_dir in real_series[:2]:
        (study_dir / series_dir.name).mkdir(parents=True)

    with pytest.raises(StudyDecodeError):
        prep_study(study_uid, tmp_path, series_df, k_slices=4, size=32, max_series=4)


@pytest.mark.skipif(not _HAS_REAL_STUDY, reason="no real downloaded study available")
def test_prep_study_records_an_unreadable_slice_and_keeps_the_rest_of_the_series(tmp_path):
    # one truncated file (a partially completed download -- there is a real
    # zero-byte one in data/sample) must cost that slice, not the series.
    series_df = pd.read_csv(_TEST_SERIES_CSV)
    study_uid = _real_study_uid()
    real_series = sorted(p for p in (_TEST_SERIES_DIR / study_uid).iterdir() if p.is_dir())[0]

    series_dir = tmp_path / study_uid / real_series.name
    shutil.copytree(real_series, series_dir)
    for stray in series_dir.glob("*.dcm"):
        if stray.stat().st_size == 0:
            stray.unlink()
    (series_dir / "truncated.dcm").write_bytes(b"")

    subset_df = series_df[series_df["SeriesInstanceUID"] == real_series.name]
    series_slices, meta = prep_study(
        study_uid, tmp_path, subset_df, k_slices=4, size=32, max_series=1
    )

    assert len(series_slices[real_series.name]) == 4
    assert [f["stage"] for f in meta["decode_failures"]] == ["header"]


@pytest.mark.skipif(not _HAS_REAL_STUDY, reason="no real downloaded study available")
def test_prep_study_output_survives_a_save_load_round_trip(tmp_path):
    series_df = pd.read_csv(_TEST_SERIES_CSV)
    study_uid = _real_study_uid()
    path = tmp_path / f"{study_uid}.npz"

    save_study_npz(path, *prep_study(study_uid, _TEST_SERIES_DIR, series_df, k_slices=4, size=32))
    loaded_slices, loaded_meta = load_study_npz(path)

    assert loaded_meta["StudyInstanceUID"] == study_uid
    assert set(loaded_slices) == set(loaded_meta["series"])
