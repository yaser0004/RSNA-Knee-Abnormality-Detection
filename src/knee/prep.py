import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from knee.dicom import (
    SLOTS,
    StudyDecodeError,
    _SERIES_PRIORITY,
    _read_slice_header,
    sequence_weighting,
    census_study_laterality,
    order_slices,
    percentile_clip_to_uint8,
    read_rescaled_pixels,
    select_k_evenly_spaced,
    select_series,
    select_slice_groups,
    select_slots,
    select_slots_v3,
)


# save_study_npz/load_study_npz use np.savez(..., dtype=object) + allow_pickle=True.
# This is safe here: every .npz this pipeline reads was written by this same
# pipeline (Kaggle prep notebooks -> a private Kaggle dataset we control), never
# loaded from a third party or the competition's own data. Pickle's arbitrary-code-
# execution risk applies to untrusted input, which this isn't.


def normalize_series(raw_slices: list[np.ndarray]) -> list[np.ndarray]:
    """Robust percentile clip (1st/99th), computed once across all the slices
    handed in rather than per slice -- this is the per-series upgrade
    decode_and_normalize's docstring calls for: a shared scale keeps relative
    intensity (fluid vs. bone) comparable across a series' slices instead of
    independently renormalizing each one.

    prep_study passes the K slices it selected, not the full series, so the
    scale describes exactly the pixels that get stored (and costs one decode
    pass, not two)."""
    if not raw_slices:
        return []
    stacked = np.concatenate([s.ravel() for s in raw_slices])
    lo, hi = np.percentile(stacked, [1, 99])
    return [percentile_clip_to_uint8(s, lo, hi) for s in raw_slices]


_KNOWN_SIDES = {"L", "R"}


def mirror_to_canonical(image: np.ndarray, side: str | None, canonical: str = "R") -> np.ndarray:
    """Flip a slice left-right when its resolved side is the other standard
    side (L/R), so four of the twelve labels (Medial/Lateral Meniscus,
    Medial/Lateral OA) don't require the model to learn laterality
    implicitly. Anything other than a recognized L/R side -- None
    (unresolved), or a real but non-mirrorable code like "B" (bilateral, both
    knees in one series -- 1 study in the Phase 2 census) -- is returned
    unmirrored rather than guessed at; the plan requires excluding studies
    the resolver can't cleanly canonicalize from laterality-dependent
    training instead of flipping blind."""
    if side in _KNOWN_SIDES and side != canonical:
        return np.fliplr(image)
    return image


def _series_attributes(series_df: pd.DataFrame, series_uid: str) -> dict:
    row = series_df[series_df["SeriesInstanceUID"] == series_uid]
    if row.empty:
        return {"Anatomical_Plane": None, "Fluid_Sensitive": None, "Fat_Suppression": None}
    row = row.iloc[0]
    return {
        "Anatomical_Plane": row.get("Anatomical_Plane"),
        "Fluid_Sensitive": int(row["Fluid_Sensitive"]) if "Fluid_Sensitive" in row else None,
        "Fat_Suppression": int(row["Fat_Suppression"]) if "Fat_Suppression" in row else None,
    }


def _pad_to_square(image: np.ndarray) -> np.ndarray:
    """Zero-letterbox a slice to a square so the following resize preserves its
    aspect ratio. 7.9% of the corpus's series are non-square (Phase 2 pilot,
    2026-08-10), worst ratio 2:1 -- resizing those straight to a square squashes
    the anatomy, and does so by a different factor per series. Padding costs a
    little effective resolution on those slices and nothing on the other 92%."""
    height, width = image.shape
    if height == width:
        return image
    side = max(height, width)
    padded = np.zeros((side, side), dtype=image.dtype)
    top = (side - height) // 2
    left = (side - width) // 2
    padded[top:top + height, left:left + width] = image
    return padded


# Below the field of view of ~99.6% of the corpus (Rows x PixelSpacing has median
# 160 mm and runs 70-320), so the crop lands inside the acquired image for nearly
# every study and the rest get padded rather than upscaled.
CROP_MM = 130.0

# 130 mm / 336 px = 0.387 mm/px. A feature of width d survives resampling only at a
# pitch <= d/2, and a meniscal tear is 1-3 mm: 224 px gives 0.580 mm/px and misses
# a 1 mm tear, 336 clears it. Artifacts are stored at 336 so the loader can hand a
# model either resolution without a second prep run.
CROP_PX = 336


def crop_to_mm(
    image: np.ndarray,
    pixel_spacing: tuple[float, float] | list[float] | None,
    crop_mm: float = CROP_MM,
    out_size: int = CROP_PX,
) -> np.ndarray:
    """Centre-crop a slice to a fixed *physical* extent and resample it to a
    fixed pixel grid, so every study reaches the model at the same mm/px.

    This replaces the fixed-pixel letterbox, which normalized no physical scale
    at all: studies in this corpus differ in mm/px by a factor of several, and
    under the old path the model saw a 3 mm feature at whatever pixel width the
    acquisition happened to give it. No downstream capacity recovers that.

    `pixel_spacing` is DICOM's [row, column] pair and the two axes are not
    always equal, so the crop takes a different pixel count on each. A field of
    view smaller than `crop_mm` is zero-padded *after* centring rather than
    upscaled -- padding costs the border, upscaling would misstate the scale
    that is the whole point here.

    Raises ValueError when spacing is missing: falling back to a fixed-pixel
    crop would silently reintroduce the defect on exactly the rows that can't
    be checked. prep skips such a series and records it instead."""
    if pixel_spacing is None:
        raise ValueError("crop_to_mm needs PixelSpacing; caller must skip a series without it")
    row_mm, col_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
    if not (row_mm > 0 and col_mm > 0):
        raise ValueError(f"non-positive PixelSpacing {pixel_spacing}")

    height, width = image.shape
    take_rows = min(height, max(1, round(crop_mm / row_mm)))
    take_cols = min(width, max(1, round(crop_mm / col_mm)))
    top = (height - take_rows) // 2
    left = (width - take_cols) // 2
    cropped = image[top:top + take_rows, left:left + take_cols]

    px_per_mm = out_size / crop_mm
    out_rows = min(out_size, max(1, round(take_rows * row_mm * px_per_mm)))
    out_cols = min(out_size, max(1, round(take_cols * col_mm * px_per_mm)))
    resized = np.array(
        Image.fromarray(cropped).resize((out_cols, out_rows), Image.BILINEAR)
    )

    if (out_rows, out_cols) == (out_size, out_size):
        return resized
    canvas = np.zeros((out_size, out_size), dtype=resized.dtype)
    r0 = (out_size - out_rows) // 2
    c0 = (out_size - out_cols) // 2
    canvas[r0:r0 + out_rows, c0:c0 + out_cols] = resized
    return canvas


def _failure(series_uid: str, stage: str, transfer_syntax: str | None, exc: Exception) -> dict:
    """One recorded read failure. The except clauses that build these are
    deliberately broad: pydicom surfaces a missing pixel-data handler as
    NotImplementedError, a truncated file as InvalidDicomError, a corrupt
    frame as anything from ValueError to AttributeError. The point is to
    census which transfer syntaxes fail across the corpus, not to predict the
    exception hierarchy -- so record the type and keep going. A study only
    fails outright when no series survives."""
    return {
        "SeriesInstanceUID": series_uid,
        "stage": stage,
        "TransferSyntaxUID": transfer_syntax,
        "error": type(exc).__name__,
    }


def _series_headers(series_dir: Path, series_uid: str, decode_failures: list[dict]) -> list[dict]:
    """Header-only read of every slice in a series directory, recording the
    ones that fail. Shared by both prep paths so their failure census means the
    same thing."""
    headers = []
    for dcm_path in sorted(series_dir.glob("*.dcm")):
        try:
            headers.append(_read_slice_header(dcm_path))
        except Exception as exc:
            decode_failures.append(_failure(series_uid, "header", None, exc))
    return headers


def study_weightings(study_dir: Path, series_uids: list[str]) -> dict[str, str]:
    """Contrast weighting of each of a study's series, from one header apiece.

    Feeds knee.dicom.select_slots_v3, whose weighting axis no delivered CSV
    carries. Derived here rather than looked up in results/weighting_census.csv
    on purpose: that census covers the training corpus only, so a lookup would
    work in every screen and then fail on the hidden test set, which is the worst
    place to discover it.

    TR/TE are acquisition parameters and constant within a series, so this reads
    the first slice and stops -- over 24,371 series that is minutes rather than
    hours, and the census measured the whole corpus at 2.5 minutes this way.

    Anything unreadable is "unknown", which is a real slot in the v3 table rather
    than a dropped series: 238 studies in this corpus have no recoverable TR/TE
    on any series at all."""
    weightings = {}
    for series_uid in series_uids:
        slices = sorted((Path(study_dir) / series_uid).glob("*.dcm"))
        weightings[series_uid] = "unknown"
        if not slices:
            continue
        try:
            header = _read_slice_header(slices[0])
        except Exception:
            continue
        weightings[series_uid] = sequence_weighting(
            header["RepetitionTime"], header["EchoTime"], header["ScanningSequence"])
    return weightings


def _decode_selected(
    selected: list[dict], series_uid: str, decode_failures: list[dict]
) -> tuple[list[np.ndarray], list[dict]]:
    """Decode pixels for already-selected slice headers, recording failures and
    keeping the headers that survived alongside their pixels."""
    raw_slices, kept_headers = [], []
    for header in selected:
        try:
            raw_slices.append(read_rescaled_pixels(str(header["path"])))
            kept_headers.append(header)
        except Exception as exc:
            decode_failures.append(_failure(series_uid, "pixels", header["TransferSyntaxUID"], exc))
    return raw_slices, kept_headers


def prep_slots(
    study_uid: str,
    dcm_root: str | Path,
    series_df: pd.DataFrame,
    n_groups: int = 5,
    crop_mm: float = CROP_MM,
    out_size: int = CROP_PX,
    is_gold: bool = False,
    slot_scheme: str = "v2",
) -> tuple[dict[str, list[np.ndarray]], dict]:
    """Phase 6 prep: one study to slot-keyed, physically-scaled pixels.

    Three differences from prep_study, all of which the 2026-09-05 recon named
    as most of the distance to the public 0.93 shelf:

    - series are assigned to named **slots** (plane x weighting) with no
      substitution, so an absent view is absent rather than duplicated, and the
      loader can hand the model an honest presence mask;
    - slices come in **contiguous groups**, which become an encoder input's
      three channels;
    - pixels are cropped to a fixed **physical** extent, so mm/px is a constant
      across the corpus instead of an accident of acquisition.

    Returns (slot_slices, meta). Only filled slots appear in slot_slices;
    meta["slots"] carries the full table with None for the empty ones, so
    "absent" and "failed to decode" stay distinguishable. Slices are stored
    unmirrored with the resolved side in meta, as in v1 -- mirroring is a
    load-time decision.

    A slot whose series has no PixelSpacing is dropped rather than cropped at
    fixed pixels: that fallback is the defect this function exists to remove,
    and it would apply invisibly to only some studies. The study fails only if
    no slot survives.

    `slot_scheme` picks the table. "v2" (the default) is plane x fat-suppression,
    six slots, and is the artifact behind the scored 0.905 submission. "v3" adds
    the weighting axis recovered from TR/TE -- nineteen slots -- after the
    2026-09-07 census found SAG_STRUCT alone carrying 1,645 T1, 1,702 PD, 1,224 T2
    and 388 GRE into one attention position. v2 stays the default until v3's
    re-baseline lands, the same discipline that kept prep v1 alive through
    Phase 6 Step 3.

    Only the *filing* changes between schemes: cropping, ordering, normalization
    and grouping are shared, so a 19-vs-6 screen cannot be confounded by a
    silent pixel change."""
    dcm_root = Path(dcm_root)
    study_dir = dcm_root / study_uid

    if slot_scheme == "v3":
        study_series = series_df[series_df["StudyInstanceUID"] == study_uid]
        weightings = study_weightings(study_dir, list(study_series["SeriesInstanceUID"]))
        slots = select_slots_v3(series_df, study_uid, weightings)
    elif slot_scheme == "v2":
        weightings = {}
        slots = select_slots(series_df, study_uid)
    else:
        raise ValueError(f"unknown slot_scheme {slot_scheme!r}; expected 'v2' or 'v3'")
    if not any(slots.values()):
        raise StudyDecodeError(f"no series metadata for study {study_uid}")

    census = census_study_laterality(study_dir)

    slot_slices: dict[str, list[np.ndarray]] = {}
    series_meta: dict[str, dict] = {}
    skipped_slots: dict[str, str] = {}
    decode_failures: list[dict] = []
    patient_sex = None

    for slot_name, series_uid in slots.items():
        if series_uid is None:
            skipped_slots[slot_name] = "no series for this plane/weighting"
            continue

        headers = _series_headers(study_dir / series_uid, series_uid, decode_failures)
        if not headers:
            skipped_slots[slot_name] = "no readable slice headers"
            continue
        if headers[0]["PixelSpacing"] is None:
            skipped_slots[slot_name] = "no PixelSpacing"
            continue

        groups = select_slice_groups(order_slices(headers), n_groups)
        selected = [header for group in groups for header in group]
        raw_slices, kept_headers = _decode_selected(selected, series_uid, decode_failures)
        if len(raw_slices) != len(selected):
            # a group with a hole in it is not three adjacent slices any more,
            # and silently shortening it would desynchronise the channels
            skipped_slots[slot_name] = "a slice group failed to decode"
            continue

        spacing = kept_headers[0]["PixelSpacing"]
        slot_slices[slot_name] = [
            crop_to_mm(s, spacing, crop_mm=crop_mm, out_size=out_size)
            for s in normalize_series(raw_slices)
        ]

        first = kept_headers[0]
        patient_sex = patient_sex or first["PatientSex"]
        series_meta[series_uid] = {
            **_series_attributes(series_df, series_uid),
            "slot": slot_name,
            "n_slices_stored": len(slot_slices[slot_name]),
            "n_groups": n_groups,
            "group_size": len(groups[0]),
            "crop_mm": crop_mm,
            "mm_per_px": crop_mm / out_size,
            "Rows": first["Rows"],
            "Columns": first["Columns"],
            "PixelSpacing": spacing,
            "fov_mm": [first["Rows"] * spacing[0], first["Columns"] * spacing[1]],
            "PhotometricInterpretation": first["PhotometricInterpretation"],
            "TransferSyntaxUID": first["TransferSyntaxUID"],
            "RepetitionTime": first["RepetitionTime"],
            "EchoTime": first["EchoTime"],
            "ScanningSequence": first["ScanningSequence"],
            "SeriesDescription": first["SeriesDescription"],
            # the recovered class this series was filed under. Recorded so a v3
            # artifact can be audited after the fact rather than taken on trust.
            # None under v2, which never asks -- distinct from "unknown", which
            # means it asked and the header could not answer.
            "weighting": weightings.get(series_uid),
        }

    if not slot_slices:
        raise StudyDecodeError(
            f"no usable slot for study {study_uid} ({len(skipped_slots)} skipped)"
        )

    return slot_slices, {
        "StudyInstanceUID": study_uid,
        "side": census["side"],
        "route": census["route"],
        "is_gold": is_gold,
        "PatientSex": patient_sex,
        "n_series_on_disk": census["n_series"],
        "slot_scheme": slot_scheme,
        "slots": slots,
        "series": series_meta,
        "skipped_slots": skipped_slots,
        "decode_failures": decode_failures,
    }


def prep_study(
    study_uid: str,
    dcm_root: str | Path,
    series_df: pd.DataFrame,
    k_slices: int = 24,
    size: int = 256,
    max_series: int = 4,
    is_gold: bool = False,
) -> tuple[dict[str, list[np.ndarray]], dict]:
    """Turn one study's raw DICOMs into the model-ready pixels Phase 2 stores:
    up to max_series series (knee.dicom.select_series' priority order), k
    evenly spaced slices each, per-series normalized to uint8 and resized to
    size x size. Returns (series_slices, meta) -- it does not write anything.

    Pure by design: Phase 6's submission notebook runs this same
    select -> order -> K-slice -> normalize path over the test set, where no
    prepped dataset exists to read from. A write-coupled version would grow a
    second, drifting copy of the pipeline inside infer.py.

    Pixels are stored unmirrored, with the resolved side/route in meta;
    mirroring to canonical happens at load time (PreppedStudyDataset) so the
    artifacts stay neutral to a laterality decision Phase 3 may still improve
    -- 48.3% of studies are unresolved today, and re-prepping 4,407 studies to
    pick up better coverage would be the expensive way to find that out.

    An unusable series (empty directory, every slice failing to decode) is
    skipped and the reason recorded rather than failing the study; only a
    study with no usable series at all raises StudyDecodeError."""
    dcm_root = Path(dcm_root)
    study_dir = dcm_root / study_uid

    series_uids = select_series(series_df, study_uid, max_series=max_series)
    if not series_uids:
        raise StudyDecodeError(f"no series metadata for study {study_uid}")

    census = census_study_laterality(study_dir)

    series_slices: dict[str, list[np.ndarray]] = {}
    series_meta: dict[str, dict] = {}
    skipped_series: dict[str, str] = {}
    decode_failures: list[dict] = []
    patient_sex = None

    for series_uid in series_uids:
        dcm_files = sorted((study_dir / series_uid).glob("*.dcm"))
        if not dcm_files:
            skipped_series[series_uid] = "no .dcm files"
            continue

        headers = _series_headers(study_dir / series_uid, series_uid, decode_failures)
        if not headers:
            skipped_series[series_uid] = "no readable slice headers"
            continue

        selected = select_k_evenly_spaced(order_slices(headers), k_slices)
        raw_slices, kept_headers = _decode_selected(selected, series_uid, decode_failures)

        if not raw_slices:
            skipped_series[series_uid] = "no decodable slices"
            continue

        resized = [
            np.array(Image.fromarray(_pad_to_square(s)).resize((size, size), Image.BILINEAR))
            for s in normalize_series(raw_slices)
        ]
        series_slices[series_uid] = resized

        first = kept_headers[0]
        patient_sex = patient_sex or first["PatientSex"]
        series_meta[series_uid] = {
            **_series_attributes(series_df, series_uid),
            "n_slices_stored": len(resized),
            "n_slices_original": len(dcm_files),
            "Rows": first["Rows"],
            "Columns": first["Columns"],
            "PixelSpacing": first["PixelSpacing"],
            "PhotometricInterpretation": first["PhotometricInterpretation"],
            "TransferSyntaxUID": first["TransferSyntaxUID"],
        }

    if not series_slices:
        raise StudyDecodeError(
            f"no usable series for study {study_uid} ({len(skipped_series)} skipped)"
        )

    meta = {
        "StudyInstanceUID": study_uid,
        "side": census["side"],
        "route": census["route"],
        "is_gold": is_gold,
        "PatientSex": patient_sex,
        # n_series_on_disk counts every series directory; series/ holds only the
        # ones actually stored -- distinct names so a later filter can't confuse them
        "n_series_on_disk": census["n_series"],
        "series": series_meta,
        "skipped_series": skipped_series,
        "decode_failures": decode_failures,
    }
    return series_slices, meta


# Which acquisition planes can be laterality-canonicalized by an in-plane flip.
# Verified against real ImageOrientationPatient direction cosines (Phase 2,
# 2026-08-10): Axial and Coronal series have a row direction dominantly along
# patient +x, so the image's horizontal axis *is* medial-lateral and np.fliplr
# turns a left knee into a right one. Sagittal series have a row direction
# dominantly along patient +y -- their horizontal axis is anterior-posterior,
# and flipping one mirrors the knee front-to-back instead of side-to-side.
# Medial-lateral on a sagittal series runs along the slice normal, i.e. the
# slice ordering, not within any single image.
_IN_PLANE_MIRRORABLE_PLANES = {"Axial", "Coronal"}


def mirrors_in_plane(anatomical_plane: str | None) -> bool:
    """Whether mirror_to_canonical is anatomically meaningful for this plane.
    An unknown plane returns False -- consistent with the rest of the pipeline,
    which leaves anything it can't resolve unmirrored rather than guessing."""
    return anatomical_plane in _IN_PLANE_MIRRORABLE_PLANES


def _encode_jpeg(slice_uint8: np.ndarray, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(slice_uint8).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _decode_jpeg(blob: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(blob)))


def save_study_npz(path: str | Path, series_slices: dict[str, list[np.ndarray]], meta: dict) -> None:
    """Pack one study's already-normalized, already-mirrored per-series
    slices into a single .npz: JPEG-encoded (q=92) to hit the plan's
    ~0.7-1.5MB/study storage budget, plus a metadata dict (side, route,
    is_gold, ...) alongside. One pickled payload rather than one npz array
    key per series -- SeriesInstanceUIDs aren't guaranteed to make clean
    array keys, and a study only has a handful of series to begin with.

    Stored uncompressed: the payload is already JPEG, so deflate over it buys
    close to nothing and costs CPU on ~423k slices at both write and read."""
    encoded = {
        series_uid: [_encode_jpeg(s) for s in slices]
        for series_uid, slices in series_slices.items()
    }
    np.savez(path, payload=np.array({"series": encoded, "meta": meta}, dtype=object))


def _priority_rank(series_uid: str, series_meta: dict) -> tuple:
    """Sort key reproducing knee.dicom._SERIES_PRIORITY's order (sagittal-
    fluid-sensitive first) over series already stored in a study's .npz. A
    series whose stored dict key doesn't match any priority entry -- no
    per-series meta was recorded, or its plane/fluid-sensitive combination
    isn't one of the four ranked ones -- sorts after every ranked series, tied
    on the UID itself for a deterministic order rather than an unstable one."""
    entry = series_meta.get(series_uid, {})
    key = (entry.get("Anatomical_Plane"), entry.get("Fluid_Sensitive"))
    try:
        rank = _SERIES_PRIORITY.index(key)
    except ValueError:
        rank = len(_SERIES_PRIORITY)
    return (rank, series_uid)


def load_study_npz(
    path: str | Path,
    max_series: int | None = None,
    max_slices: int | None = None,
    order: list[str] | None = None,
) -> tuple[dict[str, list[np.ndarray]], dict]:
    """Read a prepped study back. With both limits None (the default) every
    stored blob is decoded.

    The limits exist because the JPEG decode is ~85% of a study's load time,
    and a caller that only wants 2 series x 16 slices should not pay to decode
    4 x 24 and discard most of it -- that is the efficiency-track submission's
    shape, and the pilot measured the difference as roughly linear in blobs
    decoded. Series are limited by the same _SERIES_PRIORITY order prep_study
    used to pick them (sagittal-fluid-sensitive first) -- stored series live in
    a plain dict keyed by SeriesInstanceUID, so without this a max_series=1
    read would return whichever series happens to sort first alphabetically,
    not the one Phase 1's single-series config actually trained on. The
    returned dict preserves this order, so PreppedStudyDataset can iterate it
    directly instead of re-deriving the order itself."""
    data = np.load(path, allow_pickle=True)
    payload = data["payload"].item()

    stored = payload["series"]
    series_meta = payload["meta"].get("series", {})
    if order is not None:
        # v2 artifacts are keyed by slot name, and the slot table *is* the
        # order -- there is nothing for _priority_rank to rank.
        ordered = [key for key in order if key in stored]
    else:
        ordered = sorted(stored, key=lambda uid: _priority_rank(uid, series_meta))
    wanted = ordered if max_series is None else ordered[:max_series]
    series_slices = {
        series_uid: [_decode_jpeg(blob) for blob in stored[series_uid][:max_slices]]
        for series_uid in wanted
    }
    return series_slices, payload["meta"]
