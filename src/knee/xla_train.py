"""Pieces of the training loop that stay valid under XLA's static-graph rules.

XLA compiles a graph per distinct shape and per distinct baked-in Python constant, so three habits
of the GPU loop each cost a recompile on every step: boolean-mask indexing (data-dependent shape),
an optimizer whose learning rate is a Python float that changes every step, and a short last
batch. These replacements keep the same arithmetic with fixed shapes and device-tensor
hyperparameters; tests pin them to the GPU loop's `masked_bce_loss`, `torch.optim.Adam` and
`OneCycleLR` on CPU.
"""
import torch
import torch.nn.functional as F


def masked_bce_loss_static(logits: torch.Tensor, targets: torch.Tensor,
                           weights: torch.Tensor | None = None) -> torch.Tensor:
    """`knee.train.masked_bce_loss` without boolean indexing: NaN targets get weight zero.

    The GPU version raises when a batch has nothing to train on; a raise needs a host read, which
    stalls the XLA graph, so an empty batch gives zero loss and zero gradient instead."""
    valid = ~torch.isnan(targets)
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    per_element = F.binary_cross_entropy_with_logits(logits, safe_targets, reduction="none")
    w = valid.to(per_element.dtype)
    if weights is not None:
        w = w * weights
    return (per_element * w).sum() / w.sum().clamp(min=1e-12)


def one_cycle_schedule(total_steps: int, max_lr: float) -> tuple[list[float], list[float]]:
    """Per-step (learning rate, Adam beta1) of `OneCycleLR` with the project's defaults.

    Taken from the real scheduler rather than re-derived, so the shape and the momentum cycle
    (0.95 -> 0.85 -> 0.95) match it by construction."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    probe = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([probe], lr=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, epochs=1, steps_per_epoch=total_steps)
    lrs, beta1s = [], []
    for _ in range(total_steps):
        lrs.append(float(opt.param_groups[0]["lr"]))
        beta1s.append(float(opt.param_groups[0]["betas"][0]))
        opt.step()
        sched.step()
    return lrs, beta1s


class TensorAdam:
    """`torch.optim.Adam` (no weight decay) with learning rate and beta1 passed as device tensors.

    A tensor argument is a graph input, so changing it every step does not recompile; a Python
    float would be baked in as a constant. The update is torch's own: lerp-free but equal to
    rounding, with the bias corrections taken from the current beta1."""

    def __init__(self, params, beta2: float = 0.999, eps: float = 1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.beta2, self.eps = beta2, eps
        self.exp_avg = [torch.zeros_like(p) for p in self.params]
        self.exp_avg_sq = [torch.zeros_like(p) for p in self.params]
        self.t = torch.zeros((), device=self.params[0].device)

    @torch.no_grad()
    def step(self, lr: torch.Tensor, beta1: torch.Tensor) -> None:
        self.t += 1
        bias1 = 1 - beta1 ** self.t
        bias2_sqrt = (1 - self.beta2 ** self.t).sqrt()
        step_size = lr / bias1
        for p, m, v in zip(self.params, self.exp_avg, self.exp_avg_sq):
            g = p.grad
            if g is None:
                continue
            m.mul_(beta1).add_(g * (1 - beta1))
            v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            denom = v.sqrt() / bias2_sqrt + self.eps
            p.sub_(step_size * m / denom)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


def pad_to_multiple(items: list, multiple: int) -> tuple[list, int]:
    """Pad with copies of the first item to a multiple of `multiple`; returns (padded, n_real).

    Evaluation batches must keep one shape, so the last short batch is filled with repeats whose
    predictions the caller drops by slicing to n_real."""
    if not items:
        raise ValueError("nothing to pad")
    short = (-len(items)) % multiple
    return list(items) + [items[0]] * short, len(items)
