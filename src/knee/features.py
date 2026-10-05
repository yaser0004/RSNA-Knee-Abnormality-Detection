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


class PooledBackbone(torch.nn.Module):
    """A timm ViT as a frozen per-image feature extractor: [N, 3, H, W] in [0, 1] ->
    [N, 2 * D], the CLS token followed by the mean of the patch tokens.

    Registers (DINOv3 has four) sit between CLS and the patches, so the patch mean starts
    at `num_prefix_tokens`, not at 1. Normalisation is done here because the pipeline's
    loader hands over raw [0, 1] pixels: the fine-tuned members learned that scale, a frozen
    backbone expects the statistics it was pretrained with."""

    def __init__(self, model: torch.nn.Module,
                 mean: tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: tuple[float, ...] = (0.229, 0.224, 0.225)):
        super().__init__()
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.num_features = 2 * int(model.num_features)
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)  # a frozen extractor never leaves eval mode

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features((x - self.mean) / self.std)
        prefix = self.model.num_prefix_tokens
        return torch.cat([tokens[:, 0], tokens[:, prefix:].mean(dim=1)], dim=-1)


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
