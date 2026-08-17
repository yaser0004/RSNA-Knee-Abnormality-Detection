import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from knee.dicom import (
    StudyDecodeError,
    _SERIES_PRIORITY,
    _read_slice_header,
    census_study_laterality,
    order_slices,
    percentile_clip_to_uint8,
    read_rescaled_pixels,
    select_k_evenly_spaced,
    select_series,
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

        headers = []
        for dcm_path in dcm_files:
            try:
                headers.append(_read_slice_header(dcm_path))
            except Exception as exc:
                decode_failures.append(_failure(series_uid, "header", None, exc))
        if not headers:
            skipped_series[series_uid] = "no readable slice headers"
            continue

        selected = select_k_evenly_spaced(order_slices(headers), k_slices)

        raw_slices = []
        kept_headers = []
        for header in selected:
            try:
                raw_slices.append(read_rescaled_pixels(str(header["path"])))
                kept_headers.append(header)
            except Exception as exc:
                decode_failures.append(
                    _failure(series_uid, "pixels", header["TransferSyntaxUID"], exc)
                )

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
    ordered = sorted(stored, key=lambda uid: _priority_rank(uid, series_meta))
    wanted = ordered if max_series is None else ordered[:max_series]
    series_slices = {
        series_uid: [_decode_jpeg(blob) for blob in stored[series_uid][:max_slices]]
        for series_uid in wanted
    }
    return series_slices, payload["meta"]
