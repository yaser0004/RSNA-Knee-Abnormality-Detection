"""Frozen-backbone feature extraction and its on-disk cache.

A prepped study is [n_slots, n_groups, 3, H, W] with a presence mask (PreppedSlotDataset).
`extract_study_features` turns it into [n_slots, n_groups, D] with a backbone that is
never trained, running only the images that exist; the cache stores those features for a
whole corpus so heads, targets and seeds can be tried without touching the images again."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class FeatureCache:
    features: np.ndarray   # [N, n_slots, n_groups, D] float16, zeros under a False mask
    mask: np.ndarray       # [N, n_slots] bool
    uids: np.ndarray       # [N] unicode, the row order every consumer must agree on
    backbone: str


@torch.no_grad()
def extract_study_features(backbone: torch.nn.Module, volume: torch.Tensor,
                           present: torch.Tensor) -> torch.Tensor:
    """[n_slots, n_groups, 3, H, W] + [n_slots] bool -> [n_slots, n_groups, D].
    Absent slots are black squares; running them wastes most of the pass (about 5 of 6
    slots are filled on average) and their features would not mean "absent", so they
    are skipped and left at exactly zero, which the head's mask then ignores."""
    if volume.dim() != 5:
        raise ValueError("extract_study_features needs [n_slots, n_groups, 3, H, W]; "
                         f"got {tuple(volume.shape)}: the flat volume has no n_slots, n_groups axes")
    n_slots, n_groups = volume.shape[:2]
    images = volume[present].reshape(-1, *volume.shape[2:])
    if images.shape[0] == 0:
        return torch.zeros(n_slots, n_groups, _width(backbone))
    out = backbone(images)
    feats = out.new_zeros(n_slots, n_groups, out.shape[-1])
    feats[present] = out.reshape(int(present.sum()), n_groups, -1)
    return feats


def _width(backbone: torch.nn.Module) -> int:
    width = getattr(backbone, "num_features", None)
    if width is None:
        raise ValueError("a study with no present slot needs backbone.num_features to size its zeros")
    return int(width)


def save_feature_cache(path: str | Path, features: np.ndarray, mask: np.ndarray,
                       uids: list[str], *, backbone: str) -> None:
    """Plain arrays, no pickle: the cache is a derived dataset and is read back with
    allow_pickle=False, so a file from outside this pipeline cannot execute anything."""
    if features.shape[0] != len(uids) or mask.shape[0] != len(uids):
        raise ValueError(f"{features.shape[0]} feature rows and {mask.shape[0]} mask rows "
                         f"for {len(uids)} uids")
    if np.any(features[~mask] != 0):
        raise ValueError("features under a False mask must be exactly zero (absent slot): a "
                         "nonzero value would leak into any unmasked pooling")
    np.savez(path, features=features.astype(np.float16), mask=mask.astype(bool),
             uids=np.asarray(uids, dtype=str), backbone=np.asarray(backbone, dtype=str))


def load_feature_cache(path: str | Path, expect_uids: list[str] | None = None) -> FeatureCache:
    with np.load(path, allow_pickle=False) as z:
        cache = FeatureCache(features=z["features"], mask=z["mask"], uids=z["uids"],
                             backbone=str(z["backbone"]))
    if expect_uids is not None and list(cache.uids) != list(expect_uids):
        raise ValueError("feature cache row order differs from the expected uid order; "
                         "a mismatched order would silently pair features with the wrong labels")
    return cache
