import io
from pathlib import Path

import numpy as np
from PIL import Image

from knee.dicom import percentile_clip_to_uint8


# save_study_npz/load_study_npz use np.savez(..., dtype=object) + allow_pickle=True.
# This is safe here: every .npz this pipeline reads was written by this same
# pipeline (Kaggle prep notebooks -> a private Kaggle dataset we control), never
# loaded from a third party or the competition's own data. Pickle's arbitrary-code-
# execution risk applies to untrusted input, which this isn't.


def normalize_series(raw_slices: list[np.ndarray]) -> list[np.ndarray]:
    """Robust percentile clip (1st/99th), computed once across every slice in
    the series rather than per slice -- this is the per-series upgrade
    decode_and_normalize's docstring calls for: a shared scale keeps relative
    intensity (fluid vs. bone) comparable across a study's slices instead of
    independently renormalizing each one."""
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
    array keys, and a study only has a handful of series to begin with."""
    encoded = {
        series_uid: [_encode_jpeg(s) for s in slices]
        for series_uid, slices in series_slices.items()
    }
    np.savez_compressed(path, payload=np.array({"series": encoded, "meta": meta}, dtype=object))


def load_study_npz(path: str | Path) -> tuple[dict[str, list[np.ndarray]], dict]:
    data = np.load(path, allow_pickle=True)
    payload = data["payload"].item()
    series_slices = {
        series_uid: [_decode_jpeg(blob) for blob in blobs]
        for series_uid, blobs in payload["series"].items()
    }
    return series_slices, payload["meta"]
