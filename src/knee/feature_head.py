"""Per-diagnosis attention head over cached backbone features.

The readout is `KneeModel._attend` -- the same function the trained group_attention
models use -- applied to features that were extracted once from a frozen backbone,
so a head trained here is comparable member-for-member with the fine-tuned ones and
costs minutes, not GPU-hours, to train. Only the window handling is new: random
window subsampling at train time was the largest single-change gain of Phase 8
(NOTES Screen A), and on a cache it is free."""
import numpy as np
import torch
import torch.nn as nn

from knee.model import KneeModel


def window_indices(n_stored: int, n_keep: int, *, random: bool,
                   rng: np.random.Generator | None = None) -> torch.Tensor:
    """Which of the `n_stored` windows a read uses. Evaluation takes the fixed spread
    (9 stored, 5 kept -> 0,2,4,6,8, the pixels the five-group models trained on);
    training draws a sorted random subset, so no fixed set can be memorised."""
    if n_keep > n_stored:
        raise ValueError(f"asked for {n_keep} windows but only {n_stored} are stored")
    if random:
        rng = rng if rng is not None else np.random.default_rng()
        idx = np.sort(rng.choice(n_stored, size=n_keep, replace=False))
    else:
        idx = np.round(np.linspace(0, n_stored - 1, n_keep)).astype(np.int64)
    return torch.as_tensor(idx, dtype=torch.long)


class FeatureHead(nn.Module):
    """[batch, n_slots, n_groups, D] cached features + [batch, n_slots] presence
    mask -> [batch, num_labels] logits. Parameters are named as in KneeModel so a
    trained model's head can be loaded directly."""

    def __init__(self, n_features: int, num_labels: int = 12):
        super().__init__()
        self.num_labels = num_labels
        self.slot_scores = nn.Linear(n_features, num_labels)
        self.label_readout = nn.Parameter(torch.empty(num_labels, n_features))
        nn.init.normal_(self.label_readout, std=n_features ** -0.5)
        self.label_bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, features: torch.Tensor, mask: torch.Tensor,
                windows: torch.Tensor | None = None) -> torch.Tensor:
        if features.dim() != 4:
            raise ValueError("FeatureHead needs [batch, n_slots, n_groups, features]; "
                             f"got {tuple(features.shape)}")
        # zero what the mask hides, so a study with no present slot reads bias-only whatever the
        # caller left under it (the trained models get this from skipping absent images)
        features = features * mask[:, :, None, None].to(features.dtype)
        if windows is not None:
            features = features[:, :, windows]
        batch, n_slots, n_groups, n_features = features.shape
        tokens = features.reshape(batch, n_slots * n_groups, n_features)
        present = mask.repeat_interleave(n_groups, dim=1)
        # an unbound call on purpose: one readout implementation for every head
        return KneeModel._attend(self, tokens, present)[0]
