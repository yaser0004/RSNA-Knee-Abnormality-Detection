from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from knee.dicom import (
    SLICE_GROUP,
    SLOTS,
    StudyDecodeError,
    _read_slice_header,
    decode_and_normalize,
    order_slices,
    select_k_evenly_spaced,
    select_series,
)
from knee.infer import LABEL_COLUMNS
from torchvision.transforms.v2 import functional as TF

from knee.prep import load_study_npz, mirror_to_canonical, mirrors_in_plane

# StudyDecodeError and _read_slice_header moved to knee.dicom (knee.prep raises
# and calls them, and prep must not import this module -- that would pull torch
# into the CPU-only prep notebooks). Re-exported here because callers and tests
# already import them from knee.dataset.
__all__ = [
    "CachedDataset",
    "augment_volume",
    "KneeStudyDataset",
    "PreppedStudyDataset",
    "StudyDecodeError",
]


# Rigid jitter only. No flip of either axis: laterality is canonicalized upstream
# (mirror_to_canonical maps every knee onto one side) and four of the twelve labels
# are medial/lateral pairs, so a horizontal flip would destroy exactly the signal
# that canonicalization exists to create.
_AUG_ROT_DEG = 8.0
_AUG_TRANSLATE = 0.05
_AUG_SCALE = 0.08
_AUG_INTENSITY = 0.10


def augment_volume(volume: torch.Tensor) -> torch.Tensor:
    """Train-time jitter for one study's stack, shape [n, H, W] in [0, 1].

    One transform is drawn per call and applied to every slice: a study is a
    single sample, and jittering slices independently would desynchronize the
    anatomy down the stack, which is noise rather than augmentation.

    Screen A measured train stable-6 0.979 against val 0.836 at 8 epochs
    (NOTES 2026-09-05), so the model is fitting its training set and this is the
    lever for that, not more capacity.

    Parameters are drawn from torch's global RNG, so seeding the run seeds the
    augmentation with it."""
    angle = float(torch.empty(()).uniform_(-_AUG_ROT_DEG, _AUG_ROT_DEG))
    scale = float(torch.empty(()).uniform_(1.0 - _AUG_SCALE, 1.0 + _AUG_SCALE))
    max_shift = _AUG_TRANSLATE * volume.shape[-1]
    shift = [float(torch.empty(()).uniform_(-max_shift, max_shift)) for _ in range(2)]
    gain = float(torch.empty(()).uniform_(1.0 - _AUG_INTENSITY, 1.0 + _AUG_INTENSITY))

    # affine() takes [..., H, W]; passing the whole stack at once is what applies
    # the identical transform to every slice
    out = TF.affine(volume.unsqueeze(1), angle=angle, translate=shift, scale=scale,
                    shear=[0.0, 0.0]).squeeze(1)
    return (out * gain).clamp_(0.0, 1.0)


def _lookup_weights(weights_df: pd.DataFrame, study_uid: str) -> torch.Tensor:
    """Per-label sample weights for one study, defaulting to 1.0.

    A study absent from the frame gets full weight, not zero: a missing row means
    the sources were never compared for it, which is not a reason to drop it from
    the loss. Zero would do exactly that, silently."""
    row = weights_df[weights_df["StudyInstanceUID"] == study_uid]
    if row.empty:
        return torch.ones(len(LABEL_COLUMNS))
    values = row[LABEL_COLUMNS].to_numpy(dtype="float32")[0]
    return torch.nan_to_num(torch.tensor(values), nan=1.0)


def _lookup_labels(labels_df: pd.DataFrame | None, study_uid: str) -> torch.Tensor | None:
    """None when the dataset carries no labels at all (inference); a row of
    NaN when this particular study has none (the weak-supervision case, where
    NaN means "excluded from the loss", never "negative")."""
    if labels_df is None:
        return None
    row = labels_df[labels_df["StudyInstanceUID"] == study_uid]
    if row.empty:
        return torch.full((len(LABEL_COLUMNS),), float("nan"))
    return torch.tensor(row[LABEL_COLUMNS].to_numpy()[0], dtype=torch.float32)


class KneeStudyDataset(Dataset):
    """Phase 1 baseline: one series per study (see knee.dicom.select_series),
    n_slices evenly spaced, decoded and normalized to [0, 1]. Directory layout
    matches the competition's own: dcm_root/<StudyUID>/<SeriesUID>/<SOPUID>.dcm."""

    def __init__(
        self,
        study_uids: list[str],
        dcm_root: str | Path,
        series_df: pd.DataFrame,
        labels_df: pd.DataFrame | None = None,
        n_slices: int = 16,
        size: int = 224,
        max_series: int = 1,
    ):
        self.study_uids = list(study_uids)
        self.dcm_root = Path(dcm_root)
        self.series_df = series_df
        self.labels_df = labels_df
        self.n_slices = n_slices
        self.size = size
        self.max_series = max_series

    def __len__(self) -> int:
        return len(self.study_uids)

    def __getitem__(self, idx: int):
        study_uid = self.study_uids[idx]
        image = self._load_image(study_uid)
        labels = self._lookup_labels(study_uid)
        return image, labels, study_uid

    def _load_image(self, study_uid: str) -> torch.Tensor:
        series_uids = select_series(self.series_df, study_uid, max_series=self.max_series)
        if not series_uids:
            raise StudyDecodeError(f"no series metadata for study {study_uid}")
        series_uid = series_uids[0]

        series_dir = self.dcm_root / study_uid / series_uid
        headers = [_read_slice_header(p) for p in sorted(series_dir.glob("*.dcm"))]
        ordered = order_slices(headers)
        selected = select_k_evenly_spaced(ordered, self.n_slices)
        if not selected:
            raise StudyDecodeError(f"no decodable slices for study {study_uid}, series {series_uid}")

        slices = [decode_and_normalize(str(h["path"]), size=self.size) for h in selected]
        while len(slices) < self.n_slices:
            slices.append(slices[-1])

        volume = np.stack(slices).astype(np.float32) / 255.0
        return torch.from_numpy(volume).unsqueeze(1).repeat(1, 3, 1, 1)

    def _lookup_labels(self, study_uid: str) -> torch.Tensor | None:
        return _lookup_labels(self.labels_df, study_uid)


class PreppedStudyDataset(Dataset):
    """Reads the Phase 2 prep artifacts (one <StudyUID>.npz per study, written
    by prep_study + save_study_npz) instead of raw DICOMs, so a training epoch
    costs a JPEG decode per slice rather than a DICOM decode. Same
    (image, labels, study_uid) contract as KneeStudyDataset, so train.py is
    unchanged.

    Mirroring to canonical happens here rather than in the artifact: the
    stored pixels stay neutral to a laterality resolution Phase 3 may still
    improve. That paid for itself once already -- the in-plane flip turned out
    to be wrong for Sagittal series (see mirrors_in_plane), and fixing it cost
    nothing because the artifacts were never mirrored. mirror_to_canonical is
    applied per slice *before* np.stack -- np.fliplr returns a negative-stride
    view and torch.from_numpy rejects those.

    The artifacts hold 1-4 series of variable length. Padding is the same
    repeat-the-last rule KneeStudyDataset already uses, extended to the series
    axis so every study collates to a fixed (max_series * n_slices) volume.
    That flattening is a Phase 2 placeholder: Phase 5's attention pooling wants
    a real series axis with a validity mask, and should replace it rather than
    inherit it."""

    def __init__(
        self,
        study_uids: list[str],
        npz_root: str | Path,
        labels_df: pd.DataFrame | None = None,
        n_slices: int = 24,
        max_series: int = 4,
        canonical: str = "R",
        sides: dict[str, str] | None = None,
        augment: bool = False,
    ):
        self.study_uids = list(study_uids)
        self.npz_root = Path(npz_root)
        self.labels_df = labels_df
        self.n_slices = n_slices
        self.max_series = max_series
        self.canonical = canonical
        # Per-study side override, keyed by StudyInstanceUID. Phase 2 wrote
        # side=None into 49% of the artifacts because no tag resolved; the
        # geometry route recovers those (NOTES 2026-09-05) and the artifacts
        # store unmirrored pixels, so the correction belongs here rather than in
        # a re-prep. A study absent from the map keeps its stored side.
        self.sides = sides or {}
        # train-time only: the val and gold tiers must see the same pixels every
        # epoch or their scores are not comparable across configs
        self.augment = augment

    def __len__(self) -> int:
        return len(self.study_uids)

    def __getitem__(self, idx: int):
        study_uid = self.study_uids[idx]
        image = self._load_image(study_uid)
        labels = self._lookup_labels(study_uid)
        return image, labels, study_uid

    def _load_image(self, study_uid: str) -> torch.Tensor:
        path = self.npz_root / f"{study_uid}.npz"
        if not path.exists():
            raise StudyDecodeError(f"no prepped artifact at {path}")

        # decode only what this configuration returns -- at 2 series x 16 slices
        # that is a third of the blobs a full read would decode and discard
        series_slices, meta = load_study_npz(
            path, max_series=self.max_series, max_slices=self.n_slices
        )
        if not series_slices:
            raise StudyDecodeError(f"prepped artifact for {study_uid} holds no series")

        side = self.sides.get(study_uid, meta.get("side"))
        series_meta = meta.get("series", {})
        blocks = []
        # load_study_npz already orders series_slices by _SERIES_PRIORITY
        # (sagittal-fluid-sensitive first) -- iterate as returned rather than
        # re-sorting by UID, which would undo that order.
        for series_uid in series_slices:
            # only Axial/Coronal have medial-lateral along the image's horizontal
            # axis; flipping a Sagittal series mirrors anterior-posterior instead
            plane = series_meta.get(series_uid, {}).get("Anatomical_Plane")
            flip_side = side if mirrors_in_plane(plane) else None
            slices = [
                mirror_to_canonical(s, flip_side, canonical=self.canonical)
                for s in series_slices[series_uid]
            ]
            while len(slices) < self.n_slices:
                slices.append(slices[-1])
            blocks.append(slices[: self.n_slices])

        while len(blocks) < self.max_series:
            blocks.append(blocks[-1])

        volume = np.stack([s for block in blocks for s in block]).astype(np.float32) / 255.0
        if self.augment:
            # augment before the channel expand: the expand is a view, and
            # transforming it would materialize a full float32 copy per study
            return augment_volume(torch.from_numpy(volume)).unsqueeze(1).expand(-1, 3, -1, -1)
        # expand, not repeat: identical values, but a view instead of a 75MB
        # float32 copy per study (max_series * n_slices * 3 * size^2). The copy
        # measured as the single largest cost in the loader -- larger than all
        # 96 JPEG decodes -- and collate has to materialize the batch anyway.
        return torch.from_numpy(volume).unsqueeze(1).expand(-1, 3, -1, -1)

    def _lookup_labels(self, study_uid: str) -> torch.Tensor | None:
        return _lookup_labels(self.labels_df, study_uid)


_SLOT_NAMES = [name for name, _, _ in SLOTS]
_SLOT_PLANES = {name: plane for name, plane, _ in SLOTS}


class PreppedSlotDataset(Dataset):
    """Reads the Phase 6 prep v2 artifacts (prep_slots + save_study_npz) and
    returns (image, mask, labels, study_uid).

    `image` is [n_slots, n_groups, 3, H, W] and `mask` is [n_slots] of bool.
    This is the replacement PreppedStudyDataset's docstring asks for: v1
    flattened every series into one long slice axis and padded by repeating the
    last series, which cannot express "this study has no coronal acquisition" at
    all -- a repeated series and a real second series look identical to the
    model. An absent slot here is zeros with its mask entry false, so a head can
    decline to attend to it.

    The channel axis holds the three *adjacent* slices of a stored group rather
    than one slice replicated three times, which is why prep stores groups.

    Artifacts are stored at 336px so resolution stays a screenable axis without
    a second corpus prep; `out_size` downsamples on the way out.

    Mirroring stays a load-time decision, as in v1, and still respects
    mirrors_in_plane: a slot's plane comes from the slot table, so sagittal
    slots are never flipped (their horizontal axis is anterior-posterior, and
    flipping one mirrors the knee front-to-back)."""

    def __init__(
        self,
        study_uids: list[str],
        npz_root: str | Path,
        labels_df: pd.DataFrame | None = None,
        n_groups: int = 5,
        out_size: int = 224,
        slots: list[str] | None = None,
        canonical: str = "R",
        sides: dict[str, str] | None = None,
        augment: bool = False,
        weights_df: pd.DataFrame | None = None,
    ):
        self.study_uids = list(study_uids)
        self.npz_root = Path(npz_root)
        self.labels_df = labels_df
        # per-label sample weights (knee.reports.confidence_from_agreement).
        # Opt-in: with none set the batch keeps its 4-tuple shape, so a config
        # screened before this existed still pairs against one screened after.
        self.weights_df = weights_df
        self.n_groups = n_groups
        self.out_size = out_size
        # a subset keeps the slot axis meaningful: Step 3's re-baseline reads 2
        # of the 6 to sit as close to the Phase 5 configuration as v2 allows
        self.slots = list(slots) if slots is not None else list(_SLOT_NAMES)
        self.canonical = canonical
        self.sides = sides or {}
        self.augment = augment

    def __len__(self) -> int:
        return len(self.study_uids)

    def __getitem__(self, idx: int):
        study_uid = self.study_uids[idx]
        image, mask = self._load_image(study_uid)
        labels = _lookup_labels(self.labels_df, study_uid)
        if self.weights_df is None:
            return image, mask, labels, study_uid
        return image, mask, labels, _lookup_weights(self.weights_df, study_uid), study_uid

    def _select_groups(self, slices: list[np.ndarray]) -> list[np.ndarray]:
        """Take n_groups groups from the ones stored, spread across the artifact's
        band rather than from its start.

        prep stores groups spanning the 0.20-0.80 band of the slice stack. A
        config reading fewer groups than were stored must sample across that
        span: taking a prefix would confound "fewer groups" with "only the
        bottom of the joint", and a screen varying n_groups would then measure
        anatomical coverage rather than the axis it names.

        ponytail: the JPEG decode still covers every stored slice, because
        load_study_npz can only truncate a prefix, not take a spread. The waste
        is zero at n_groups=5 (all of them) and the decode is ~85% of load time,
        so this is only worth revisiting if a config actually reads fewer."""
        stored = len(slices) // SLICE_GROUP
        if stored >= self.n_groups:
            picks = np.linspace(0, stored - 1, num=self.n_groups).round().astype(int)
            return [slices[g * SLICE_GROUP + j] for g in picks for j in range(SLICE_GROUP)]

        wanted = self.n_groups * SLICE_GROUP
        padded = list(slices)
        while len(padded) < wanted:
            padded.append(padded[-1])
        return padded[:wanted]

    def _load_image(self, study_uid: str) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.npz_root / f"{study_uid}.npz"
        if not path.exists():
            raise StudyDecodeError(f"no prepped artifact at {path}")

        stored, meta = load_study_npz(path, order=self.slots)
        if not stored:
            raise StudyDecodeError(f"prepped artifact for {study_uid} holds no slot")

        side = self.sides.get(study_uid, meta.get("side"))
        size = self.out_size
        # an absent slot is zeros at whatever the artifact's stored resolution
        # is, not at out_size -- they are stacked together and resized once
        stored_shape = next(iter(stored.values()))[0].shape
        blocks = []
        present = []
        for name in self.slots:
            slices = stored.get(name)
            if not slices:
                blocks.append(np.zeros((self.n_groups, SLICE_GROUP, *stored_shape), np.float32))
                present.append(False)
                continue

            flip_side = side if mirrors_in_plane(_SLOT_PLANES[name]) else None
            slices = [mirror_to_canonical(s, flip_side, canonical=self.canonical)
                      for s in slices]
            slices = self._select_groups(slices)
            arr = np.stack(slices).astype(np.float32) / 255.0
            blocks.append(arr.reshape(self.n_groups, SLICE_GROUP, *arr.shape[-2:]))
            present.append(True)

        volume = torch.from_numpy(np.stack(blocks))
        if volume.shape[-1] != size or volume.shape[-2] != size:
            flat = volume.reshape(-1, *volume.shape[-2:]).unsqueeze(1)
            volume = TF.resize(flat, [size, size], antialias=True).reshape(
                len(self.slots), self.n_groups, SLICE_GROUP, size, size)
        if self.augment:
            # one transform per study, applied to every slot and channel: a
            # study is a single sample, and jittering slots independently would
            # desynchronise the anatomy between planes
            shape = volume.shape
            volume = augment_volume(volume.reshape(-1, size, size)).reshape(shape)
        return volume, torch.tensor(present, dtype=torch.bool)


class CachedDataset(Dataset):
    """Wraps another dataset, decoding each item at most once and reusing the
    result on every later access. DICOM decode dominates runtime far more
    than a model forward/backward pass, so re-decoding every epoch would make
    multi-epoch training needlessly slow -- decode once, train many times."""

    def __init__(self, base_dataset):
        self.base = base_dataset
        self._cache: dict[int, object] = {}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        if idx not in self._cache:
            self._cache[idx] = self.base[idx]
        return self._cache[idx]
