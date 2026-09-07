from pathlib import Path

import pandas as pd
import pytest

from knee.dicom import (
    SLOTS,
    SLOTS_V3,
    order_slices,
    select_k_evenly_spaced,
    select_series,
    select_slice_groups,
    select_slots,
    select_slots_v3,
    sequence_weighting,
)

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


# --- Phase 6 prep v2: slot selection and contiguous slice groups ----------------


def _slot_series_df(rows):
    return pd.DataFrame(
        [{"StudyInstanceUID": "S", "SeriesInstanceUID": uid,
          "Anatomical_Plane": plane, "Fluid_Sensitive": fluid,
          "Fat_Suppression": fluid} for uid, plane, fluid in rows]
    )


def test_select_slots_fills_every_slot_a_study_has_a_series_for():
    df = _slot_series_df([
        ("a", "Sagittal", 1), ("b", "Coronal", 1), ("c", "Axial", 1),
        ("d", "Sagittal", 0), ("e", "Coronal", 0), ("f", "Axial", 0),
    ])

    slots = select_slots(df, "S")

    assert list(slots) == [name for name, _, _ in SLOTS]
    assert slots == {"SAG_FLUID": "a", "COR_FLUID": "b", "AX_FLUID": "c",
                     "SAG_STRUCT": "d", "COR_STRUCT": "e", "AX_STRUCT": "f"}


def test_select_slots_leaves_an_unmatched_slot_empty_rather_than_substituting():
    # No substitution is the point of the presence mask: a slot filled from a
    # different plane would assert the model is looking at anatomy it isn't,
    # and the head would divide attention across two copies of one acquisition.
    df = _slot_series_df([("a", "Sagittal", 1), ("b", "Coronal", 1)])

    slots = select_slots(df, "S")

    assert slots["SAG_FLUID"] == "a" and slots["COR_FLUID"] == "b"
    assert slots["AX_FLUID"] is None
    assert all(slots[name] is None for name in ("SAG_STRUCT", "COR_STRUCT", "AX_STRUCT"))


def test_select_slots_picks_deterministically_when_several_series_match():
    df = _slot_series_df([("z", "Sagittal", 1), ("a", "Sagittal", 1), ("m", "Sagittal", 1)])

    assert select_slots(df, "S")["SAG_FLUID"] == "a"


def test_select_slots_never_puts_one_series_in_two_slots():
    df = _slot_series_df([("a", "Sagittal", 1), ("b", "Sagittal", 0)])

    filled = [uid for uid in select_slots(df, "S").values() if uid is not None]

    assert len(filled) == len(set(filled))


def test_select_slots_returns_all_slots_empty_for_an_unknown_study():
    df = _slot_series_df([("a", "Sagittal", 1)])

    assert set(select_slots(df, "OTHER").values()) == {None}


def _headers(n):
    return [{"InstanceNumber": i, "path": f"{i}.dcm"} for i in range(n)]


def test_select_slice_groups_returns_contiguous_triples_inside_the_central_band():
    # 3 adjacent slices become the encoder's 3 channels, so they have to be
    # physically adjacent -- evenly spaced singletons cannot express that. The
    # band skips the outermost slices, which are mostly soft tissue outside
    # the joint.
    groups = select_slice_groups(_headers(30), n_groups=4)

    assert len(groups) == 4
    for group in groups:
        idx = [h["InstanceNumber"] for h in group]
        assert idx == list(range(idx[0], idx[0] + 3))
        assert 5 <= idx[0] and idx[-1] <= 25


def test_select_slice_groups_spreads_groups_across_the_band():
    groups = select_slice_groups(_headers(40), n_groups=3)

    centres = [g[1]["InstanceNumber"] for g in groups]
    assert centres == sorted(centres)
    assert centres[-1] - centres[0] > 10


def test_select_slice_groups_clamps_on_a_stack_shorter_than_the_group():
    # A one-slice series still has to produce a full group rather than raise:
    # the alternative is dropping the series, and a thin series is not an
    # unusable one.
    groups = select_slice_groups(_headers(1), n_groups=2)

    assert len(groups) == 2
    assert all(len(g) == 3 for g in groups)
    assert all(h["InstanceNumber"] == 0 for g in groups for h in g)


def test_select_slice_groups_never_runs_off_either_end():
    for n in range(1, 12):
        for g in select_slice_groups(_headers(n), n_groups=3):
            assert all(0 <= h["InstanceNumber"] < n for h in g)
            assert len(g) == 3


class TestSequenceWeighting:
    """Recovering T1 / PD / T2 / GRE from TR, TE and ScanningSequence.

    train_series.csv's Fluid_Sensitive and Fat_Suppression are identical on all
    24,371 rows, so as delivered they carry one axis and a slot table built from
    them cannot separate T1 from non-fat-suppressed PD/T2. This is the census
    route that says whether that conflation costs anything -- pilkwang's default
    slot scheme separates them and Phase 6 shipped their fallback.

    Thresholds are the standard knee-MRI partition: short TR is T1, long TR with
    short TE is proton-density, long TR with long TE is T2. Deliberately no
    SeriesDescription parsing -- it is free text and the hidden test set is not
    guaranteed to phrase it the way training does.
    """

    def test_short_tr_is_t1(self):
        assert sequence_weighting("500", "12", None) == "T1"

    def test_long_tr_short_te_is_proton_density(self):
        assert sequence_weighting("3000", "30", "SE") == "PD"

    def test_long_tr_long_te_is_t2(self):
        assert sequence_weighting("4000", "80", "SE") == "T2"

    def test_gradient_echo_is_reported_separately(self):
        # a GRE sequence's TR/TE do not carry the spin-echo meaning, so the
        # partition above would mislabel it rather than merely miss it
        assert sequence_weighting("600", "15", "GR") == "GRE"
        assert sequence_weighting("600", "15", "['GR', 'IR']") == "GRE"

    @pytest.mark.parametrize("tr, te", [(None, "12"), ("500", None), (None, None),
                                        ("", "12"), ("not-a-number", "12")])
    def test_missing_or_unparseable_tags_are_unknown_never_guessed(self, tr, te):
        # a fixed-pixel-style silent fallback is the failure mode this project has
        # already paid for once; an explicit bucket is countable, a guess is not
        assert sequence_weighting(tr, te, "SE") == "unknown"

    def test_boundaries_are_closed_consistently(self):
        assert sequence_weighting("800", "12", "SE") == "PD"   # TR at the bound
        assert sequence_weighting("799", "12", "SE") == "T1"
        assert sequence_weighting("3000", "40", "SE") == "T2"  # TE at the bound
        assert sequence_weighting("3000", "39", "SE") == "PD"

    def test_every_result_is_one_of_the_declared_classes(self):
        classes = {"T1", "PD", "T2", "GRE", "unknown"}
        for tr in (None, "300", "800", "5000"):
            for te in (None, "5", "40", "120"):
                for seq in (None, "SE", "GR", "IR"):
                    assert sequence_weighting(tr, te, seq) in classes


class TestSlotsV3:
    """The recovered slot table: plane x fat-suppression x weighting.

    The census (2026-09-07) established that train_series.csv's one axis encodes
    fat suppression while TR/TE encode weighting, and that they are orthogonal --
    SAG_STRUCT alone held 1,645 T1, 1,702 PD, 1,224 T2 and 388 GRE arriving at one
    attention position. v3 splits that axis out.

    SLOTS (v2) stays until the v3 re-baseline lands, because it is the artifact
    behind the scored 0.905 submission -- the same discipline that kept prep v1
    alive through Phase 6 Step 3.
    """

    def _series(self, rows):
        return pd.DataFrame(
            [{"StudyInstanceUID": "s", "SeriesInstanceUID": uid,
              "Anatomical_Plane": plane, "Fluid_Sensitive": fs} for uid, plane, fs in rows])

    def test_the_table_covers_every_plane_and_carries_an_unknown_tier(self):
        planes = {plane for _, plane, _, _ in SLOTS_V3}
        assert planes == {"Sagittal", "Coronal", "Axial"}
        # the tier that keeps the 238 header-stripped studies in the corpus
        assert {w for _, _, _, w in SLOTS_V3} == {"T1", "PD", "T2", "UNK"}
        assert len({name for name, _, _, _ in SLOTS_V3}) == len(SLOTS_V3)

    def test_a_series_lands_in_the_slot_for_its_plane_weighting_and_suppression(self):
        df = self._series([("a", "Sagittal", 0)])
        slots = select_slots_v3(df, "s", {"a": "T1"})
        assert slots["SAG_NOFS_T1"] == "a"
        assert slots["SAG_NOFS_PD"] is None

    def test_an_unrecoverable_weighting_goes_to_the_unknown_tier_never_a_guess(self):
        """The 238-study case: a whole study of header-stripped series. A silent
        assignment to the plane's dominant weighting would be the fixed-pixel
        defect again -- wrong on exactly the rows nobody can check."""
        df = self._series([("a", "Axial", 1)])
        slots = select_slots_v3(df, "s", {"a": "unknown"})
        assert slots["AXI_FS_UNK"] == "a"
        assert all(uid is None for name, uid in slots.items() if name != "AXI_FS_UNK")

    def test_a_series_with_no_weighting_recorded_at_all_is_also_unknown(self):
        df = self._series([("a", "Axial", 1)])
        assert select_slots_v3(df, "s", {})["AXI_FS_UNK"] == "a"

    def test_a_dropped_combination_lands_nowhere_rather_than_being_rehomed(self):
        # GRE is excluded by the 10% rule; it must not be folded into T1 or T2
        df = self._series([("a", "Sagittal", 0)])
        assert all(uid is None for uid in select_slots_v3(df, "s", {"a": "GRE"}).values())

    def test_no_series_can_occupy_two_slots(self):
        df = self._series([("a", "Coronal", 1), ("b", "Coronal", 0), ("c", "Axial", 1)])
        slots = select_slots_v3(df, "s", {"a": "PD", "b": "T1", "c": "T2"})
        filled = [uid for uid in slots.values() if uid is not None]
        assert sorted(filled) == ["a", "b", "c"]

    def test_ties_break_on_the_uid_so_shard_order_cannot_change_the_artifact(self):
        rows = [("zzz", "Coronal", 1), ("aaa", "Coronal", 1)]
        weighting = {"zzz": "PD", "aaa": "PD"}
        first = select_slots_v3(self._series(rows), "s", weighting)
        second = select_slots_v3(self._series(rows[::-1]), "s", weighting)
        assert first == second
        assert first["COR_FS_PD"] == "aaa"

    def test_other_studies_series_are_not_selected(self):
        df = self._series([("a", "Coronal", 1)])
        df.loc[0, "StudyInstanceUID"] = "other"
        assert all(uid is None for uid in select_slots_v3(df, "s", {"a": "PD"}).values())
