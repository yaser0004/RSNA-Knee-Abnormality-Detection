import numpy as np
import pytest

from knee.prep import (
    load_study_npz,
    mirror_to_canonical,
    normalize_series,
    save_study_npz,
)


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
