import math

import pytest
import torch

from knee.train import masked_bce_loss
from knee.xla_train import TensorAdam, masked_bce_loss_static, one_cycle_schedule, pad_to_multiple


def _batch(seed=0, nan_rate=0.25):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(8, 12, generator=g)
    targets = torch.rand(8, 12, generator=g)
    targets[torch.rand(8, 12, generator=g) < nan_rate] = float("nan")
    weights = torch.rand(8, 12, generator=g) + 0.1
    return logits, targets, weights


def test_static_loss_equals_the_masked_loss_with_and_without_weights():
    logits, targets, weights = _batch()
    assert torch.allclose(masked_bce_loss_static(logits, targets), masked_bce_loss(logits, targets), atol=1e-6)
    assert torch.allclose(masked_bce_loss_static(logits, targets, weights),
                          masked_bce_loss(logits, targets, weights), atol=1e-6)


def test_static_loss_gradient_equals_and_nan_targets_get_none():
    logits, targets, _ = _batch(1)
    a, b = logits.clone().requires_grad_(), logits.clone().requires_grad_()
    masked_bce_loss(a, targets).backward()
    masked_bce_loss_static(b, targets).backward()
    assert torch.allclose(a.grad, b.grad, atol=1e-6)
    assert (b.grad[torch.isnan(targets)] == 0).all()


def test_static_loss_on_an_all_nan_batch_is_zero_not_nan():
    logits = torch.randn(4, 12, requires_grad=True)
    loss = masked_bce_loss_static(logits, torch.full((4, 12), float("nan")))
    loss.backward()
    assert loss.item() == 0.0 and torch.isfinite(logits.grad).all() and (logits.grad == 0).all()


def test_one_cycle_schedule_starts_low_peaks_and_cycles_momentum():
    lrs, beta1s = one_cycle_schedule(total_steps=100, max_lr=3e-4)
    assert len(lrs) == len(beta1s) == 100
    assert lrs[0] == pytest.approx(3e-4 / 25)
    assert max(lrs) == pytest.approx(3e-4, rel=1e-3) and lrs.index(max(lrs)) == pytest.approx(29, abs=2)
    assert lrs[-1] < lrs[0]
    assert min(beta1s) == pytest.approx(0.85, abs=1e-3) and max(beta1s) == pytest.approx(0.95, abs=1e-3)


def test_tensor_adam_follows_torch_adam_with_a_one_cycle_schedule():
    torch.manual_seed(0)
    ref = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))
    mine = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))
    mine.load_state_dict(ref.state_dict())
    steps = 30
    opt = torch.optim.Adam(ref.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, epochs=1, steps_per_epoch=steps)
    lrs, beta1s = one_cycle_schedule(steps, 3e-4)
    adam = TensorAdam(mine.parameters())
    x, y = torch.randn(16, 6), torch.randn(16, 3)
    for i in range(steps):
        for model in (ref, mine):
            model.zero_grad()
            torch.nn.functional.mse_loss(model(x), y).backward()
        opt.step()
        sched.step()
        adam.step(lr=torch.tensor(lrs[i]), beta1=torch.tensor(beta1s[i]))
    for a, b in zip(ref.parameters(), mine.parameters()):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)


def test_tensor_adam_skips_parameters_without_gradients_and_checks_inputs():
    used, unused = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(3))
    adam = TensorAdam([used, unused])
    (used * 2).sum().backward()
    adam.step(lr=torch.tensor(0.1), beta1=torch.tensor(0.9))
    assert not torch.equal(used, torch.ones(3)) and torch.equal(unused, torch.ones(3))
    with pytest.raises(ValueError):
        one_cycle_schedule(total_steps=0, max_lr=1e-3)


def test_pad_to_multiple_keeps_order_and_reports_how_many_are_real():
    padded, n_real = pad_to_multiple(["a", "b", "c", "d", "e"], 4)
    assert padded == ["a", "b", "c", "d", "e", "a", "a", "a"] and n_real == 5
    assert pad_to_multiple(["a", "b"], 2) == (["a", "b"], 2)
    with pytest.raises(ValueError):
        pad_to_multiple([], 4)
    assert math.isfinite(1.0)
