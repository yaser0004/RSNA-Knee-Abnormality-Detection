from pathlib import Path

import pytest

from knee.dicom import (
    census_study_laterality,
    image_centre_x,
    read_laterality_header,
    resolve_laterality,
    resolve_study_laterality,
    side_from_geometry,
)

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
            "geometry",
            "geometry_midline",
        }
        assert result["n_series"] == len(result["slice_counts"])
        assert all(count >= 0 for count in result["slice_counts"])


# --- geometry route (Phase 5): image CENTRE, not the corner ------------------
# The corner form sign(IPP[0]) was measured against the real tag and rejected
# (see test_does_not_guess_from_ipp_sign_when_tags_absent, which still stands).
# These cover the centre form, which is a different rule: IPP is the first
# voxel, up to half a field of view from the anatomy, so on a knee scanned near
# the midline the corner sits on the far side of x=0 from the knee itself.

def _geom_header(px, *, iop=(1, 0, 0, 0, 1, 0), spacing=(0.5, 0.5), rows=256, cols=256):
    """A slice whose row direction is +x, so centre_x = px + 0.5 * cols/2."""
    return {
        "ImagePositionPatient": [px, -50.0, 100.0],
        "ImageOrientationPatient": list(iop),
        "PixelSpacing": list(spacing),
        "Rows": rows,
        "Columns": cols,
    }


def test_image_centre_x_offsets_the_corner_by_half_the_field_of_view():
    # 256 columns at 0.5 mm/px puts the centre 64 mm along +x from the corner
    assert image_centre_x(_geom_header(-100.0)) == pytest.approx(-36.0)


def test_image_centre_x_is_none_without_the_geometry_tags():
    header = _geom_header(-100.0)
    header["ImageOrientationPatient"] = None

    assert image_centre_x(header) is None


def test_geometry_resolves_right_knee_from_negative_centre():
    side, route = side_from_geometry([_geom_header(-200.0)])

    assert side == "R"
    assert route == "geometry"


def test_geometry_resolves_left_knee_from_positive_centre():
    side, route = side_from_geometry([_geom_header(0.0)])

    assert side == "L"
    assert route == "geometry"


def test_geometry_reads_left_where_the_corner_sign_would_say_right():
    """The regression this whole route exists for. A left knee scanned near the
    midline has a negative corner x and a positive centre x -- the corner form
    calls it R, the centre form calls it L, and the tag says L."""
    header = _geom_header(-17.1)

    assert header["ImagePositionPatient"][0] < 0        # corner says R
    assert image_centre_x(header) > 0                   # centre says L
    assert side_from_geometry([header])[0] == "L"


def test_geometry_refuses_to_guess_inside_the_midline_band():
    # centre lands at -4 mm, well inside the band where the sign is chance
    side, route = side_from_geometry([_geom_header(-68.0)])

    assert side is None
    assert route == "geometry_midline"


def test_geometry_is_unavailable_when_no_series_carries_the_tags():
    header = _geom_header(-200.0)
    header["PixelSpacing"] = None

    side, route = side_from_geometry([header])

    assert side is None
    assert route == "geometry_unavailable"


def test_geometry_takes_the_median_across_a_studys_series():
    """One series with broken geometry must not drag the study off its side."""
    headers = [_geom_header(-200.0), _geom_header(-210.0), _geom_header(300.0)]

    assert side_from_geometry(headers)[0] == "R"


def test_study_laterality_falls_back_to_geometry_when_no_tag_resolves():
    """The 49% of studies with no usable tag. Without this the submission path
    preps test studies with side=None and never mirrors them, while training
    used corrected sides -- a train/test mismatch on four of the twelve labels."""
    headers = [{**_header(), **_geom_header(-200.0)}]

    side, route = resolve_study_laterality(headers)

    assert side == "R"
    assert route == "geometry"


def test_study_laterality_prefers_a_real_tag_over_geometry():
    """Geometry disagrees with the tag on 0.9% of tagged studies, so it must
    stay strictly a fallback -- the tag is the only ground truth there is."""
    headers = [{**_header(Laterality="L"), **_geom_header(-200.0)}]

    side, route = resolve_study_laterality(headers)

    assert side == "L"
    assert route == "Laterality"


def test_study_laterality_uses_geometry_when_tags_conflict():
    """Conflicting tags mean no tag can be trusted, which is the same situation
    as having none -- geometry is independent evidence, not a tie-break."""
    headers = [{**_header(Laterality="L"), **_geom_header(-200.0)},
               {**_header(Laterality="R"), **_geom_header(-200.0)}]

    side, route = resolve_study_laterality(headers)

    assert side == "R"
    assert route == "geometry"


def test_study_laterality_stays_unknown_when_geometry_is_midline_too():
    headers = [{**_header(), **_geom_header(-68.0)}]

    side, route = resolve_study_laterality(headers)

    assert side is None
    assert route == "geometry_midline"


def test_laterality_header_carries_the_geometry_fields():
    """resolve_study_laterality reads both tag and geometry off the same header,
    so the header reader has to supply both or the fallback silently never fires."""
    dcm = next((_TRAIN_SERIES_DIR).rglob("*.dcm"))

    header = read_laterality_header(dcm)

    for field in ("ImagePositionPatient", "ImageOrientationPatient",
                  "PixelSpacing", "Rows", "Columns"):
        assert field in header, field
    assert image_centre_x(header) is not None
