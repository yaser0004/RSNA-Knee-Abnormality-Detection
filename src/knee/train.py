import torch
import torch.nn.functional as F


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE over only the non-NaN targets. Lexical/pseudo-labels leave a study's
    label NaN when no rule matches (see knee.reports.lexical_label) -- that
    means "no evidence", not "negative", so it must never enter the loss."""
    mask = ~torch.isnan(targets)
    if not mask.any():
        raise ValueError("all targets in this batch are NaN -- nothing to train on")
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask])
