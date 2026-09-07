import re
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image


class StudyDecodeError(Exception):
    """Raised when a study's series or slices can't be resolved -- no series
    metadata, a missing study directory, or every candidate series unusable
    (e.g. a partially completed download). A single named exception from one
    shared code path rather than an ambiguous IndexError leaking out of list
    indexing, so callers (knee.infer.build_submission's per-study fallback, a
    training loop's collate step, or the Phase 2 prep shard loop) can catch it
    deliberately. Lives here rather than in knee.dataset because knee.prep
    raises it too, and prep must not import dataset -- that would pull torch
    into the CPU-only prep notebooks and make the two modules circular."""


# Priority order established in the plan's Phase 2 spec: sagittal-fluid-sensitive,
# coronal-fluid-sensitive, axial-fluid-sensitive, sagittal-non-fluid.
_SERIES_PRIORITY = [
    ("Sagittal", 1),
    ("Coronal", 1),
    ("Axial", 1),
    ("Sagittal", 0),
]


def select_series(series_df: pd.DataFrame, study_uid: str, max_series: int = 4) -> list[str]:
    study_series = series_df[series_df["StudyInstanceUID"] == study_uid]

    selected: list[str] = []
    used = set()
    for plane, fluid_sensitive in _SERIES_PRIORITY[:max_series]:
        match = study_series[
            (study_series["Anatomical_Plane"] == plane)
            & (study_series["Fluid_Sensitive"] == fluid_sensitive)
        ]
        for series_uid in match["SeriesInstanceUID"]:
            if series_uid not in used:
                selected.append(series_uid)
                used.add(series_uid)
                break

    if len(selected) < max_series:
        for series_uid in study_series["SeriesInstanceUID"]:
            if len(selected) >= max_series:
                break
            if series_uid not in used:
                selected.append(series_uid)
                used.add(series_uid)

    return selected


def sequence_weighting(
    repetition_time: str | None,
    echo_time: str | None,
    scanning_sequence: str | None,
) -> str:
    """Contrast weighting of one series, recovered from its header: "T1", "PD",
    "T2", "GRE" or "unknown".

    Exists because train_series.csv cannot express this. Its Fluid_Sensitive and
    Fat_Suppression columns are identical on all 24,371 rows, so SLOTS collapses
    T1 and non-fat-suppressed PD/T2 into one _STRUCT slot per plane -- two
    different tissue contrasts arriving at one attention position. Whether that
    costs anything was a counting question; this is what counted it, and
    SLOTS_V3 is what the count led to.

    The partition is the standard knee-MRI one: short TR is T1, long TR with a
    short TE is proton-density, long TR with a long TE is T2. Gradient echo is
    identified from ScanningSequence and reported separately, because its TR/TE
    do not carry the spin-echo meaning and the thresholds would mislabel it
    rather than merely miss it.

    No SeriesDescription parsing, deliberately: it is free text, and a rule fitted
    to how the training set phrases it has no guarantee on the hidden test set --
    the same argument that kept SLOTS on the delivered CSV columns.

    A missing or unparseable TR/TE returns "unknown" rather than a default. A
    silent fallback here would be the fixed-pixel defect again: wrong on only the
    rows nobody can check, and invisible in the aggregate."""
    if scanning_sequence:
        tokens = re.split(r"[^A-Za-z]+", scanning_sequence.upper())
        if "GR" in tokens:
            return "GRE"

    try:
        tr = float(repetition_time)
        te = float(echo_time)
    except (TypeError, ValueError):
        return "unknown"

    if tr < _TR_T1_MAX_MS:
        return "T1"
    return "T2" if te >= _TE_T2_MIN_MS else "PD"


# Thresholds in milliseconds. Standard knee-MRI values rather than tuned ones:
# tuning them against our own corpus would fit the training distribution, and
# this rule has to hold on the hidden test set too.
_TR_T1_MAX_MS = 800.0
_TE_T2_MIN_MS = 40.0


# Six input slots: three acquisition planes crossed with the one weighting axis
# train_series.csv actually carries. Fluid_Sensitive and Fat_Suppression are
# identical on all 24,371 rows (checked 2026-09-06), so as delivered they are one
# column under two names, and a slot table cannot separate T1 from
# non-fat-suppressed PD/T2 without recovering TR/TE/ScanningSequence from the
# headers.
# ponytail: CSV-only slots, no header parsing -- works for the hidden test set
# from the same columns and cannot fail on an unusual acquisition. Ceiling: a
# _STRUCT slot mixes T1 with non-suppressed PD/T2, which carry different tissue
# contrast. That ceiling was measured on 2026-09-07 and it is real: SAG_STRUCT
# holds 1,645 T1, 1,702 PD, 1,224 T2 and 388 GRE. SLOTS_V3 below is the upgrade.
# This table stays until v3's re-baseline lands -- it is the artifact behind the
# scored 0.905 submission, the same reason prep v1 outlived prep v2's arrival.
SLOTS = [
    ("SAG_FLUID", "Sagittal", 1),
    ("COR_FLUID", "Coronal", 1),
    ("AX_FLUID", "Axial", 1),
    ("SAG_STRUCT", "Sagittal", 0),
    ("COR_STRUCT", "Coronal", 0),
    ("AX_STRUCT", "Axial", 0),
]


def select_slots(series_df: pd.DataFrame, study_uid: str) -> dict[str, str | None]:
    """Map each slot to one of the study's series, or None when the study has
    no series for it.

    Deliberately no substitution: an empty slot stays empty and the presence
    mask carries that, because filling a missing plane from a different one
    would put a single acquisition in two slots and let the model divide its
    attention across two copies of it while the mask claimed two views. This
    is the difference from select_series, which ranks and backfills.

    Slot predicates partition the series (a series has exactly one plane and
    one Fluid_Sensitive value), so no series can land in two slots. Ties are
    broken on the UID so a study prepped in two different shard orders gives
    the same artifact."""
    study_series = series_df[series_df["StudyInstanceUID"] == study_uid]
    slots: dict[str, str | None] = {}
    for name, plane, fluid_sensitive in SLOTS:
        match = study_series[
            (study_series["Anatomical_Plane"] == plane)
            & (study_series["Fluid_Sensitive"] == fluid_sensitive)
        ]
        uids = sorted(match["SeriesInstanceUID"])
        slots[name] = uids[0] if uids else None
    return slots


# v3 slot table: plane x fat-suppression x weighting, from the 2026-09-07 census.
#
# The Phase 6 reading -- "Fluid_Sensitive and Fat_Suppression are identical on all
# 24,371 rows, so the CSV carries one axis" -- was a correct observation with the
# wrong conclusion. The CSV column encodes *fat suppression*; TR/TE encode
# *weighting*; they are orthogonal, and SLOTS above uses one of them. The cost was
# measured: SAG_STRUCT alone carried 1,645 T1, 1,702 PD, 1,224 T2 and 388 GRE into
# a single attention position, with no consistent tissue contrast there to route to.
#
# Membership rule: every combination at >=10% study fill, plus every _UNK
# combination regardless of fill. GRE (764 series) is excluded -- its TR/TE follow
# neither the T1 nor the T2 rule and folding it into either would be a silent
# mislabel, and filing it under UNK would put a positively-identified class into
# the residual bucket. It costs 19 studies (0.4%) a whole plane and none of them
# all their slots. Revisit only if the 19-vs-6 screen shows more positions help.
#
# THE _UNK TIER IS LOAD-BEARING, not padding. 1,206 series (4.9%) carry no
# recoverable TR/TE, and 238 studies (5.4%) consist *entirely* of such series --
# complete 5-to-7-series exams whose headers were stripped. Under a weighting-only
# table those 238 get zero slots: dropped from training and answered with the 0.5
# fallback at inference. No threshold fixes that; the bucket has to exist. It is
# also what contains this table's one regression against v2 -- v2 needed no header
# parsing and so could not be surprised by the hidden test set, whereas here an
# unreadable test series lands in a slot the model has actually been trained on
# rather than in a guess.
SLOTS_V3 = [
    ("SAG_FS_PD", "Sagittal", 1, "PD"),
    ("SAG_FS_T2", "Sagittal", 1, "T2"),
    ("SAG_NOFS_T1", "Sagittal", 0, "T1"),
    ("SAG_NOFS_PD", "Sagittal", 0, "PD"),
    ("SAG_NOFS_T2", "Sagittal", 0, "T2"),
    ("COR_FS_PD", "Coronal", 1, "PD"),
    ("COR_FS_T2", "Coronal", 1, "T2"),
    ("COR_NOFS_T1", "Coronal", 0, "T1"),
    ("COR_NOFS_T2", "Coronal", 0, "T2"),
    ("AXI_FS_PD", "Axial", 1, "PD"),
    ("AXI_FS_T2", "Axial", 1, "T2"),
    ("AXI_NOFS_T1", "Axial", 0, "T1"),
    ("AXI_NOFS_T2", "Axial", 0, "T2"),
    ("SAG_FS_UNK", "Sagittal", 1, "UNK"),
    ("SAG_NOFS_UNK", "Sagittal", 0, "UNK"),
    ("COR_FS_UNK", "Coronal", 1, "UNK"),
    ("COR_NOFS_UNK", "Coronal", 0, "UNK"),
    ("AXI_FS_UNK", "Axial", 1, "UNK"),
    ("AXI_NOFS_UNK", "Axial", 0, "UNK"),
]


def select_slots_v3(
    series_df: pd.DataFrame,
    study_uid: str,
    weighting_by_series: dict[str, str],
) -> dict[str, str | None]:
    """select_slots over SLOTS_V3, with the weighting axis supplied per series.

    Weighting cannot come from the delivered CSV, so the caller reads one header
    per series and passes sequence_weighting's answer.

    UNK means one thing: the weighting could not be recovered. A series the caller
    never classified counts as that. **GRE does not**, and is dropped instead --
    it is a contrast we positively identified and chose not to model, so filing it
    under "unknown" would put a known class into the residual bucket and rebuild
    the conflation this table exists to remove. The cost was measured before the
    table was fixed: 764 series, 19 studies (0.4%) losing a whole plane, none
    losing all their slots.

    Empty slots stay empty and the presence mask carries that -- no substitution,
    exactly as in select_slots."""
    study_series = series_df[series_df["StudyInstanceUID"] == study_uid]
    slots: dict[str, str | None] = {}
    for name, plane, fluid_sensitive, weighting in SLOTS_V3:
        match = study_series[
            (study_series["Anatomical_Plane"] == plane)
            & (study_series["Fluid_Sensitive"] == fluid_sensitive)
        ]
        uids = sorted(
            uid for uid in match["SeriesInstanceUID"]
            if _slot_weighting(weighting_by_series.get(uid)) == weighting
        )
        slots[name] = uids[0] if uids else None
    return slots


def _slot_weighting(recovered: str | None) -> str:
    """Map sequence_weighting's answer onto the slot table's weighting axis.
    T1/PD/T2 pass through and an unrecoverable weighting becomes UNK. "GRE" is
    returned unchanged, which matches no slot -- see select_slots_v3."""
    if recovered in ("T1", "PD", "T2", "GRE"):
        return recovered
    return "UNK"


# The outermost slices of a knee series are mostly soft tissue outside the joint,
# so groups are spread over a central band rather than the whole stack.
_SLICE_BAND = (0.20, 0.80)
SLICE_GROUP = 3


def select_slice_groups(
    ordered_headers: list[dict],
    n_groups: int,
    group_size: int = SLICE_GROUP,
    band: tuple[float, float] = _SLICE_BAND,
) -> list[list[dict]]:
    """Pick n_groups runs of `group_size` *physically adjacent* slices, spread
    across the central band of an already-ordered stack.

    The adjacency is the point: the three slices become one encoder input's
    three channels, which gives it local through-plane context a single slice
    replicated three times cannot. select_k_evenly_spaced cannot express this --
    its picks are maximally far apart by construction.

    A stack shorter than the group is clamped rather than rejected: indices run
    off the ends onto the edge slice, so a one-slice series still yields full
    groups. A thin series is not an unusable one."""
    n = len(ordered_headers)
    if n == 0:
        raise ValueError("select_slice_groups needs at least one slice")
    lo = min(max(0, round(band[0] * n)), n - 1)
    hi = min(max(lo, round(band[1] * n)), n - 1)
    centres = np.linspace(lo, hi, num=n_groups) if n_groups > 1 else [(lo + hi) / 2]

    half = group_size // 2
    groups = []
    for centre in centres:
        start = int(round(centre)) - half
        groups.append([
            ordered_headers[min(max(start + offset, 0), n - 1)]
            for offset in range(group_size)
        ])
    return groups


def _slice_normal_projection(header: dict) -> float | None:
    ipp = header.get("ImagePositionPatient")
    iop = header.get("ImageOrientationPatient")
    if ipp is None or iop is None:
        return None
    row = np.array(iop[:3], dtype=float)
    col = np.array(iop[3:], dtype=float)
    normal = np.cross(row, col)
    return float(np.dot(np.array(ipp, dtype=float), normal))


def order_slices(headers: list[dict]) -> list[dict]:
    """Sort slice headers along the slice normal, falling back to InstanceNumber
    when ImagePositionPatient/ImageOrientationPatient aren't both available."""
    projections = [_slice_normal_projection(h) for h in headers]
    if all(p is not None for p in projections):
        return [h for _, h in sorted(zip(projections, headers), key=lambda pair: pair[0])]
    return sorted(headers, key=lambda h: h["InstanceNumber"])


def select_k_evenly_spaced(ordered_headers: list[dict], k: int) -> list[dict]:
    """Pick up to k evenly spaced headers from an already-ordered slice list,
    always including the first and last, without repeating an index."""
    n = len(ordered_headers)
    if n <= k:
        return list(ordered_headers)
    indices = np.linspace(0, n - 1, num=k)
    seen = set()
    result = []
    for idx in indices:
        i = int(round(idx))
        if i not in seen:
            seen.add(i)
            result.append(ordered_headers[i])
    if len(result) != k:
        # spacing between consecutive linspace points exceeds 1 whenever n > k, so
        # rounded indices should never collide -- fail loudly if that ever breaks
        # instead of silently handing callers a shorter batch than they asked for.
        raise AssertionError(f"expected {k} distinct slices, got {len(result)} (n={n})")
    return result


# DICOM's Laterality/ImageLaterality are CS (code string) tags, standard values
# R/L/B(ilateral)/U(npaired) -- but the real corpus (Phase 2 census, 2026-08-09)
# has 20 studies using spelled-out "RIGHT"/"LEFT" instead of the single-letter
# code. Normalize so a caller comparing side to "R"/"L" (e.g. prep.py's
# mirror_to_canonical) doesn't silently mistreat a spelled-out tag as a
# different, unrecognized side.
_SIDE_ALIASES = {"RIGHT": "R", "LEFT": "L"}


def _normalize_side(raw: str) -> str:
    upper = raw.strip().upper()
    return _SIDE_ALIASES.get(upper, upper)


def resolve_laterality(header: dict) -> tuple[str | None, str]:
    """Resolve L/R for a study, in the order: ImageLaterality tag -> Laterality
    tag -> SeriesDescription/BodyPartExamined string match. Returns (side, route).

    No ImagePositionPatient-sign fallback: verified against 20 real studies (see
    NOTES.md) that sign(IPP[0]) disagrees with the Laterality tag on 6/15 (40%) of
    comparable cases -- IPP is the corner of the first pixel relative to a
    knee-centered coil FOV, not a reliable proxy for body-relative left/right.
    Guessing from it would poison far more than the 2-3% the plan budgets for, so
    an unresolved study falls through to unknown rather than a wrong guess.

    That still holds for the corner. The *centre*-based route is a different
    rule and lives in side_from_geometry, which resolve_study_laterality applies
    only after every tag route here has failed."""
    if header.get("ImageLaterality"):
        return _normalize_side(header["ImageLaterality"]), "ImageLaterality"

    if header.get("Laterality"):
        return _normalize_side(header["Laterality"]), "Laterality"

    for field in ("SeriesDescription", "BodyPartExamined"):
        text = header.get(field)
        if text and re.search(r"right", text, re.IGNORECASE):
            return "R", field
        if text and re.search(r"left", text, re.IGNORECASE):
            return "L", field

    return None, "unknown"


# Inside this distance of the midline the sign of the image centre is no better
# than chance, so a study centred there stays unresolved rather than guessed.
_MIDLINE_BAND_MM = 20.0


def image_centre_x(header: dict) -> float | None:
    """Patient-relative x of the image CENTRE, in mm, or None if the geometry
    tags are absent.

    ImagePositionPatient is the first voxel -- a corner, up to half a field of
    view from the anatomy -- so its sign is not the side the knee is on. On a
    knee scanned near the midline the corner lands on the far side of x=0 from
    the knee itself, which is precisely how the corner form (still asserted
    against in test_does_not_guess_from_ipp_sign_when_tags_absent) got it wrong.
    Walking to the centre first removes that offset:

        c = p + r * col_spacing * (Columns / 2) + d * row_spacing * (Rows / 2)

    with r and d the row and column direction cosines of
    ImageOrientationPatient. +x points to the patient's left."""
    ipp = header.get("ImagePositionPatient")
    iop = header.get("ImageOrientationPatient")
    spacing = header.get("PixelSpacing")
    rows, cols = header.get("Rows"), header.get("Columns")
    if not ipp or not iop or not spacing or not rows or not cols:
        return None

    row_spacing, col_spacing = float(spacing[0]), float(spacing[1])
    # iop[:3] is the direction of increasing COLUMN index, iop[3:] of increasing
    # row index -- the pairing with PixelSpacing is crossed, and getting it
    # backwards is silent on the square images this corpus mostly holds.
    return float(ipp[0]
                 + iop[0] * col_spacing * (cols / 2.0)
                 + iop[3] * row_spacing * (rows / 2.0))


def side_from_geometry(series_headers: list[dict]) -> tuple[str | None, str]:
    """Resolve a study's side from slice geometry alone, for the ~49% of studies
    carrying no Laterality/ImageLaterality tag at all.

    The median over a study's series is thresholded, not a single series: one
    series with corrupt geometry should not decide the study. Returns
    (None, "geometry_midline") inside the band where the sign is chance, and
    (None, "geometry_unavailable") when no series carries usable geometry --
    two different reasons to abstain, and a caller measuring coverage needs to
    tell them apart."""
    centres = [c for c in (image_centre_x(h) for h in series_headers) if c is not None]
    if not centres:
        return None, "geometry_unavailable"

    centre = statistics.median(centres)
    if abs(centre) < _MIDLINE_BAND_MM:
        return None, "geometry_midline"
    return ("L" if centre > 0 else "R"), "geometry"


def resolve_study_laterality(series_headers: list[dict]) -> tuple[str | None, str]:
    """Resolve a study's laterality from one representative header per series,
    each resolved independently via resolve_laterality. If every series that
    resolves agrees, that's the study's side (route = whichever tag produced
    it). If resolved series disagree, returns ('conflict') rather than a
    majority vote -- the plan flagged majority-vote as untested against real
    data, and a knee study shouldn't have two different sides across series,
    so a disagreement is more likely a bad tag on one series than a real
    majority to trust. See NOTES.md Phase 2 laterality census."""
    resolved = [resolve_laterality(h) for h in series_headers]
    sides = {side for side, _ in resolved if side is not None}
    if len(sides) == 1:
        side = sides.pop()
        route = next(route for s, route in resolved if s == side)
        return side, route

    # No trustworthy tag: either none resolved, or they contradict each other,
    # which are the same situation for a consumer that has to pick a side.
    # Geometry is independent evidence rather than a tie-break -- measured over
    # all 4,407 studies it agrees with the real tag on 99.08% (2162/2182) and
    # resolves 47.4% of the corpus that carried no tag at all (NOTES 2026-09-05).
    # It stays strictly a fallback: on the 0.9% where the two disagree the tag
    # wins, because the tag is the only ground truth there is.
    geom_side, geom_route = side_from_geometry(series_headers)
    if geom_route != "geometry_unavailable":
        return geom_side, geom_route

    # Geometry could not be read either, so report why the *tags* failed rather
    # than overwriting it with "geometry_unavailable" -- the census counts these
    # routes, and "the tags contradicted each other" and "there were no tags"
    # are different data problems.
    return None, ("conflict" if len(sides) > 1 else "unknown")


def read_laterality_header(dcm_path: Path) -> dict:
    """Header-only read (stop_before_pixels) of the four tags
    resolve_laterality inspects. Separate from _read_slice_header, which reads
    the geometry and acquisition fields the prep pipeline needs -- different
    consumer, different fields, not worth merging into one over-general
    reader."""
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    spacing = getattr(ds, "PixelSpacing", None)
    return {
        "ImageLaterality": getattr(ds, "ImageLaterality", None),
        "Laterality": getattr(ds, "Laterality", None),
        "SeriesDescription": getattr(ds, "SeriesDescription", None),
        "BodyPartExamined": getattr(ds, "BodyPartExamined", None),
        # geometry too: resolve_study_laterality falls back to side_from_geometry
        # off this same header, and reading it in a second pass would mean two
        # readers to keep in step for one decision
        "ImagePositionPatient": [float(x) for x in ipp] if ipp is not None else None,
        "ImageOrientationPatient": [float(x) for x in iop] if iop is not None else None,
        "PixelSpacing": [float(x) for x in spacing] if spacing is not None else None,
        "Rows": int(getattr(ds, "Rows", 0)) or None,
        "Columns": int(getattr(ds, "Columns", 0)) or None,
    }


def _plain_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _read_slice_header(dcm_path: Path) -> dict:
    """Header-only read (stop_before_pixels) of everything downstream needs
    about one slice: the geometry order_slices sorts on, and the acquisition
    tags Phase 2's prep census counts.

    The census fields live here rather than in a second pass because
    read_rescaled_pixels opens the dataset and discards it -- and a separate
    sweep would describe files rather than the slices prep actually decoded.
    TransferSyntaxUID lives on file_meta (absent in raw/implicit-VR files, so
    it's guarded); PhotometricInterpretation catches MONOCHROME1, which is
    tone-inverted and which percentile_clip_to_uint8 does not correct for."""
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    spacing = getattr(ds, "PixelSpacing", None)
    file_meta = getattr(ds, "file_meta", None)
    return {
        "SOPInstanceUID": ds.SOPInstanceUID,
        "ImagePositionPatient": [float(x) for x in ipp] if ipp is not None else None,
        "ImageOrientationPatient": [float(x) for x in iop] if iop is not None else None,
        "InstanceNumber": int(getattr(ds, "InstanceNumber", 0)),
        "Rows": int(getattr(ds, "Rows", 0)) or None,
        "Columns": int(getattr(ds, "Columns", 0)) or None,
        "PixelSpacing": [float(x) for x in spacing] if spacing is not None else None,
        # plain str, not pydicom's str subclasses: these end up pickled into the
        # prepped .npz meta, and the artifacts should be readable without pydicom
        "PhotometricInterpretation": _plain_str(getattr(ds, "PhotometricInterpretation", None)),
        "TransferSyntaxUID": _plain_str(getattr(file_meta, "TransferSyntaxUID", None)),
        "PatientSex": _plain_str(getattr(ds, "PatientSex", None)),
        # Weighting tags. Nothing reads these yet: the slot table (SLOTS) runs off
        # train_series.csv, whose Fluid_Sensitive and Fat_Suppression columns are
        # identical and so cannot separate T1 from non-fat-suppressed PD/T2.
        # Recorded into prep's per-series meta on a header read that already
        # happens, so that split can be measured later without re-prepping 4,407
        # studies just to obtain the tags.
        "RepetitionTime": _plain_str(getattr(ds, "RepetitionTime", None)),
        "EchoTime": _plain_str(getattr(ds, "EchoTime", None)),
        "ScanningSequence": _plain_str(getattr(ds, "ScanningSequence", None)),
        "SeriesDescription": _plain_str(getattr(ds, "SeriesDescription", None)),
        "path": dcm_path,
    }


def census_study_laterality(study_dir: Path) -> dict:
    """One header-only pass over a study directory
    (dcm_root/<StudyUID>/<SeriesUID>/*.dcm): resolves the study's laterality
    from one representative (first, sorted) file per series, and counts
    slices per series along the way -- Phase 2's gate needs both the
    laterality-coverage number and the slice-count distribution, and both
    come from the same directory walk, so one function answers both rather
    than scanning the corpus twice."""
    if not study_dir.is_dir():
        # a shard loop catching StudyDecodeError should also catch a study
        # directory that simply isn't there, rather than an OSError from iterdir
        raise StudyDecodeError(f"no study directory at {study_dir}")
    series_dirs = sorted(p for p in study_dir.iterdir() if p.is_dir())
    series_headers = []
    slice_counts = []
    for series_dir in series_dirs:
        dcm_files = sorted(series_dir.glob("*.dcm"))
        slice_counts.append(len(dcm_files))
        if dcm_files:
            series_headers.append(read_laterality_header(dcm_files[0]))
    side, route = resolve_study_laterality(series_headers)
    return {
        "n_series": len(series_dirs),
        "slice_counts": slice_counts,
        "side": side,
        "route": route,
        # per-series (side, route) before the study-level agreement/conflict
        # check collapses them -- lets a caller measure route-vs-route
        "series_resolutions": [resolve_laterality(h) for h in series_headers],
    }


def read_rescaled_pixels(dcm_path: str) -> np.ndarray:
    """Read one DICOM slice's pixels as float32, with RescaleSlope/Intercept
    applied -- the shared first step behind both decode_and_normalize's
    per-slice clip and prep.py's per-series clip, which needs the raw
    rescaled values from every slice in a series before it can compute a
    single series-wide percentile."""
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    return arr * slope + intercept


def percentile_clip_to_uint8(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi > lo:
        clipped = np.clip((arr - lo) / (hi - lo), 0, 1)
    else:
        clipped = np.zeros_like(arr)
    return (clipped * 255).astype(np.uint8)


def decode_and_normalize(dcm_path: str, size: int = 224) -> np.ndarray:
    """Read a single DICOM slice's pixels, robust-normalize (1st/99th
    percentile clip) to uint8, and resize. Clips per-slice rather than
    per-series -- per-series clipping needs every slice of the series loaded
    together, which this single-file entry point doesn't have; prep.py's
    normalize_series does the per-series version for the full pipeline."""
    arr = read_rescaled_pixels(dcm_path)
    lo, hi = np.percentile(arr, [1, 99])
    arr_uint8 = percentile_clip_to_uint8(arr, lo, hi)
    image = Image.fromarray(arr_uint8).resize((size, size), Image.BILINEAR)
    return np.array(image)
