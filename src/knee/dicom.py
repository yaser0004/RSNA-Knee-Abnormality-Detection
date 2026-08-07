import re

import numpy as np
import pandas as pd

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


def resolve_laterality(header: dict) -> tuple[str | None, str]:
    """Resolve L/R for a study, in the order: ImageLaterality tag -> Laterality
    tag -> SeriesDescription/BodyPartExamined string match. Returns (side, route).

    No ImagePositionPatient-sign fallback: verified against 20 real studies (see
    NOTES.md) that sign(IPP[0]) disagrees with the Laterality tag on 6/15 (40%) of
    comparable cases -- IPP is the corner of the first pixel relative to a
    knee-centered coil FOV, not a reliable proxy for body-relative left/right.
    Guessing from it would poison far more than the 2-3% the plan budgets for, so
    an unresolved study falls through to unknown rather than a wrong guess."""
    if header.get("ImageLaterality"):
        return header["ImageLaterality"], "ImageLaterality"

    if header.get("Laterality"):
        return header["Laterality"], "Laterality"

    for field in ("SeriesDescription", "BodyPartExamined"):
        text = header.get(field)
        if text and re.search(r"right", text, re.IGNORECASE):
            return "R", field
        if text and re.search(r"left", text, re.IGNORECASE):
            return "L", field

    return None, "unknown"
