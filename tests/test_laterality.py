from pathlib import Path

import pytest

from knee.dicom import census_study_laterality, resolve_laterality, resolve_study_laterality

_TRAIN_SERIES_DIR = Path(__file__).resolve().parents[1] / "data" / "sample" / "train_series"


def _header(**overrides):
    base = {
        "ImageLaterality": None,
        "Laterality": None,
        "ImagePositionPatient": None,
        "SeriesDescription": None,
        "BodyPartExamined": None,
    }
    base.update(overrides)
    return base


def test_prefers_image_laterality_over_everything_else():
    header = _header(ImageLaterality="R", Laterality="L")

    side, route = resolve_laterality(header)

    assert side == "R"
    assert route == "ImageLaterality"


def test_uses_laterality_when_image_laterality_absent():
    header = _header(Laterality="L")

    side, route = resolve_laterality(header)

    assert side == "L"
    assert route == "Laterality"


def test_does_not_guess_from_ipp_sign_when_tags_absent():
    # Verified against 20 real studies (see NOTES.md): sign(ImagePositionPatient[0])
    # disagreed with the Laterality tag on 6/15 comparable cases (40%) -- IPP is the
    # corner of the first pixel relative to a knee-centered coil FOV, not a reliable
    # proxy for body-relative left/right. Guessing from it would be actively wrong
    # more often than the plan's "2-3% silently poisons four labels" budget allows,
    # so this route is deliberately absent: unresolved falls through to unknown.
    header = _header(ImagePositionPatient=[19.4, -140.3, -7.4])

    side, route = resolve_laterality(header)

    assert side is None
    assert route == "unknown"


def test_normalizes_spelled_out_laterality_values():
    # real corpus data (Phase 2 census, 2026-08-09): 20 studies use "RIGHT"/"LEFT"
    # instead of DICOM's standard single-letter R/L code string.
    header = _header(Laterality="RIGHT")

    side, route = resolve_laterality(header)

    assert side == "R"
    assert route == "Laterality"


def test_falls_back_to_series_description_string_match():
    header = _header(SeriesDescription="knee_right_sag_pd")

    side, route = resolve_laterality(header)

    assert side == "R"
    assert route == "SeriesDescription"


def test_returns_unknown_when_nothing_resolves():
    header = _header()

    side, route = resolve_laterality(header)

    assert side is None
    assert route == "unknown"


def test_study_laterality_agrees_across_series():
    series_headers = [
        _header(ImageLaterality="R"),
        _header(Laterality="R"),
        _header(SeriesDescription="knee_right_sag_pd"),
    ]

    side, route = resolve_study_laterality(series_headers)

    assert side == "R"
    assert route == "ImageLaterality"


def test_study_laterality_ignores_unresolved_series_when_others_agree():
    series_headers = [_header(ImageLaterality="L"), _header()]

    side, route = resolve_study_laterality(series_headers)

    assert side == "L"
    assert route == "ImageLaterality"


def test_study_laterality_flags_conflict_rather_than_guessing_majority():
    series_headers = [_header(ImageLaterality="L"), _header(Laterality="R")]

    side, route = resolve_study_laterality(series_headers)

    assert side is None
    assert route == "conflict"


def test_study_laterality_does_not_spuriously_conflict_on_spelled_out_variant():
    # "R" and "RIGHT" are the same side once normalized -- must not read as a conflict.
    series_headers = [_header(Laterality="R"), _header(Laterality="RIGHT")]

    side, route = resolve_study_laterality(series_headers)

    assert side == "R"
    assert route == "Laterality"


def test_study_laterality_unknown_when_no_series_resolves():
    series_headers = [_header(), _header()]

    side, route = resolve_study_laterality(series_headers)

    assert side is None
    assert route == "unknown"


@pytest.mark.skipif(not _TRAIN_SERIES_DIR.exists(), reason="real train_series/ not downloaded")
def test_census_resolves_every_real_sample_study_by_a_named_route():
    study_dirs = sorted(p for p in _TRAIN_SERIES_DIR.iterdir() if p.is_dir())
    assert study_dirs, "expected at least one study directory in the sample"

    for study_dir in study_dirs:
        result = census_study_laterality(study_dir)
        assert result["route"] in {
            "ImageLaterality",
            "Laterality",
            "SeriesDescription",
            "BodyPartExamined",
            "conflict",
            "unknown",
        }
        assert result["n_series"] == len(result["slice_counts"])
        assert all(count >= 0 for count in result["slice_counts"])
