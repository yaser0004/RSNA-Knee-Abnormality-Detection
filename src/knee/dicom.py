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
