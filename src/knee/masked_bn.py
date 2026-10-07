"""BatchNorm whose training statistics ignore selected images, for static-shape (XLA) training.

KneeModel packs absent slots out of the backbone so BatchNorm never averages in a black square.
Boolean-mask packing gives a data-dependent shape, which XLA recompiles on every batch. Running
the backbone on every image and weighting BatchNorm's statistics by presence is the same
computation on the present images, with a fixed shape.

The swap is by class, not by module, so parameters, buffers and state_dict keys are unchanged: a
checkpoint trained this way loads into an ordinary KneeModel for GPU inference.
"""
from contextlib import contextmanager

import torch
from timm.layers.norm_act import BatchNormAct2d


class MaskedBatchNormAct2d(BatchNormAct2d):
    """timm's BatchNormAct2d, with per-image weights on the batch statistics while training.

    `_image_weights` is [N] (1.0 present, 0.0 absent) or None. With None, or in eval mode, this is
    the parent's forward unchanged."""

    _image_weights = None

    def forward(self, x):
        weights = self._image_weights
        if weights is None or not self.training:
            return super().forward(x)

        n, _, height, width = x.shape
        w = weights.to(x.dtype if x.dtype == torch.float32 else torch.float32).view(n, 1, 1, 1)
        xf = x.float()
        count = weights.float().sum() * (height * width)
        safe = count.clamp(min=1.0)
        mean = (xf * w).sum(dim=(0, 2, 3)) / safe
        centred = xf - mean.view(1, -1, 1, 1)
        var = (centred * centred * w).sum(dim=(0, 2, 3)) / safe

        if self.track_running_stats:
            momentum = 0.1 if self.momentum is None else self.momentum
            # F.batch_norm updates running_var with the unbiased estimate
            unbiased = var * safe / (safe - 1.0).clamp(min=1.0)
            has_images = (count > 0).to(mean.dtype)
            with torch.no_grad():
                self.running_mean.mul_(1 - momentum * has_images).add_(mean.detach() * momentum * has_images)
                self.running_var.mul_(1 - momentum * has_images).add_(unbiased.detach() * momentum * has_images)
                self.num_batches_tracked.add_(has_images.long())

        out = centred * torch.rsqrt(var + self.eps).view(1, -1, 1, 1)
        if self.weight is not None:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        out = self.drop(out.to(x.dtype))
        return self.act(out)


def convert_masked_bn(module: torch.nn.Module) -> int:
    """Swap every BatchNormAct2d under `module` for the masked class; returns how many."""
    swapped = 0
    for m in module.modules():
        if type(m) is BatchNormAct2d:
            m.__class__ = MaskedBatchNormAct2d
            swapped += 1
    return swapped


@contextmanager
def image_weights(module: torch.nn.Module, weights: torch.Tensor):
    """Set the per-image weights on every masked BatchNorm under `module` for the block."""
    layers = [m for m in module.modules() if isinstance(m, MaskedBatchNormAct2d)]
    for m in layers:
        m._image_weights = weights
    try:
        yield
    finally:
        for m in layers:
            m._image_weights = None
