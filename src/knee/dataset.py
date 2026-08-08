from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

from knee.dicom import decode_and_normalize, order_slices, select_k_evenly_spaced, select_series
from knee.infer import LABEL_COLUMNS


class StudyDecodeError(Exception):
    """Raised when a study's series or slices can't be resolved -- no series
    metadata, or a series directory with no .dcm files (e.g. a partially
    completed download). A single named exception from one shared code path
    rather than an ambiguous IndexError leaking out of list indexing, so
    callers (knee.infer.build_submission's per-study fallback, or a training
    loop's collate step) can catch it deliberately."""


def _read_slice_header(dcm_path: Path) -> dict:
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    return {
        "SOPInstanceUID": ds.SOPInstanceUID,
        "ImagePositionPatient": [float(x) for x in ipp] if ipp is not None else None,
        "ImageOrientationPatient": [float(x) for x in iop] if iop is not None else None,
        "InstanceNumber": int(getattr(ds, "InstanceNumber", 0)),
        "path": dcm_path,
    }


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
        if self.labels_df is None:
            return None
        row = self.labels_df[self.labels_df["StudyInstanceUID"] == study_uid]
        if row.empty:
            return torch.full((len(LABEL_COLUMNS),), float("nan"))
        return torch.tensor(row[LABEL_COLUMNS].to_numpy()[0], dtype=torch.float32)


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
