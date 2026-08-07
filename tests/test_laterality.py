from knee.dicom import resolve_laterality


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
