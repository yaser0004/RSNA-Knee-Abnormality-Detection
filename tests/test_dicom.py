from pathlib import Path

import pandas as pd
import pytest

from knee.dicom import order_slices, select_k_evenly_spaced, select_series

_TRAIN_SERIES_CSV = Path(__file__).resolve().parents[1] / "data" / "sample" / "train_series.csv"


def _series_row(study, series, plane, fluid_sensitive):
    return {
        "StudyInstanceUID": study,
        "SeriesInstanceUID": series,
        "Anatomical_Plane": plane,
        "Fluid_Sensitive": fluid_sensitive,
        "Fat_Suppression": fluid_sensitive,
    }


def test_selects_preferred_planes_in_priority_order():
    df = pd.DataFrame(
        [
            _series_row("study1", "sag_fs", "Sagittal", 1),
            _series_row("study1", "cor_fs", "Coronal", 1),
            _series_row("study1", "ax_fs", "Axial", 1),
            _series_row("study1", "sag_nf", "Sagittal", 0),
        ]
    )

    selected = select_series(df, "study1", max_series=4)

    assert selected == ["sag_fs", "cor_fs", "ax_fs", "sag_nf"]


def test_falls_back_to_any_series_when_no_priority_plane_matches():
    df = pd.DataFrame(
        [
            _series_row("study1", "oblique_a", "Oblique", 0),
            _series_row("study1", "oblique_b", "Oblique", 1),
        ]
    )

    selected = select_series(df, "study1", max_series=4)

    assert len(selected) >= 1
    assert set(selected).issubset({"oblique_a", "oblique_b"})


def test_skips_missing_plane_without_padding_with_wrong_plane():
    df = pd.DataFrame(
        [
            _series_row("study1", "sag_fs", "Sagittal", 1),
            _series_row("study1", "sag_nf", "Sagittal", 0),
        ]
    )

    selected = select_series(df, "study1", max_series=4)

    assert selected == ["sag_fs", "sag_nf"]


def test_tops_up_toward_max_series_when_priority_list_only_partially_matches():
    # only one priority slot matches (coronal-fluid-sensitive), but three more
    # series exist and max_series=4 -- all of them should come back, not just 1.
    df = pd.DataFrame(
        [
            _series_row("study1", "cor_fs", "Coronal", 1),
            _series_row("study1", "ax_nf_1", "Axial", 0),
            _series_row("study1", "ax_nf_2", "Axial", 0),
            _series_row("study1", "ax_nf_3", "Axial", 0),
        ]
    )

    selected = select_series(df, "study1", max_series=4)

    assert selected[0] == "cor_fs"
    assert len(selected) == 4
    assert len(set(selected)) == 4


@pytest.mark.skipif(not _TRAIN_SERIES_CSV.exists(), reason="real train_series.csv not downloaded")
def test_every_real_study_yields_at_least_one_series():
    df = pd.read_csv(_TRAIN_SERIES_CSV)

    for study_uid in df["StudyInstanceUID"].unique():
        selected = select_series(df, study_uid, max_series=4)
        assert len(selected) >= 1, f"{study_uid} got no series"
        assert len(selected) == len(set(selected)), f"{study_uid} got duplicate series"


def _slice_header(sop, ipp, iop, instance_number):
    return {
        "SOPInstanceUID": sop,
        "ImagePositionPatient": ipp,
        "ImageOrientationPatient": iop,
        "InstanceNumber": instance_number,
    }


def test_orders_by_ipp_projection_onto_slice_normal():
    # axial-ish orientation: row = +x, col = +y -> slice normal = +z
    iop = [1, 0, 0, 0, 1, 0]
    headers = [
        _slice_header("c", [0, 0, 10], iop, 3),
        _slice_header("a", [0, 0, 0], iop, 1),
        _slice_header("b", [0, 0, 5], iop, 2),
    ]

    ordered = order_slices(headers)

    assert [h["SOPInstanceUID"] for h in ordered] == ["a", "b", "c"]


def test_selects_k_distinct_slices_despite_duplicate_instance_numbers():
    # a study with duplicate InstanceNumber (e.g. multi-echo) but distinct
    # ImagePositionPatient/SOPInstanceUID per slice must still yield K distinct slices
    iop = [1, 0, 0, 0, 1, 0]
    headers = [_slice_header(f"sop{i}", [0, 0, i], iop, 1) for i in range(10)]

    ordered = order_slices(headers)
    selected = select_k_evenly_spaced(ordered, k=4)

    sop_uids = [h["SOPInstanceUID"] for h in selected]
    assert len(sop_uids) == 4
    assert len(set(sop_uids)) == 4
    assert sop_uids[0] == "sop0"
    assert sop_uids[-1] == "sop9"


def test_select_k_returns_all_when_fewer_slices_than_k():
    iop = [1, 0, 0, 0, 1, 0]
    headers = [_slice_header(f"sop{i}", [0, 0, i], iop, i) for i in range(3)]

    ordered = order_slices(headers)
    selected = select_k_evenly_spaced(ordered, k=8)

    assert len(selected) == 3


def test_falls_back_to_instance_number_when_position_and_orientation_missing():
    headers = [
        _slice_header("c", None, None, 3),
        _slice_header("a", None, None, 1),
        _slice_header("b", None, None, 2),
    ]

    ordered = order_slices(headers)

    assert [h["SOPInstanceUID"] for h in ordered] == ["a", "b", "c"]
