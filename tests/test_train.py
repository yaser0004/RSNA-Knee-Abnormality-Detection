import torch

from knee.train import masked_bce_loss


def test_masked_bce_loss_ignores_nan_targets():
    logits = torch.tensor([[2.0, -1.0, 0.5]], requires_grad=True)
    targets = torch.tensor([[1.0, float("nan"), 0.0]])

    loss = masked_bce_loss(logits, targets)

    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_masked_bce_loss_matches_plain_bce_when_no_nans_present():
    import torch.nn.functional as F

    logits = torch.tensor([[2.0, -1.0, 0.5]])
    targets = torch.tensor([[1.0, 0.0, 0.0]])

    expected = F.binary_cross_entropy_with_logits(logits, targets)
    actual = masked_bce_loss(logits, targets)

    assert torch.allclose(actual, expected)


def test_masked_bce_loss_raises_when_batch_is_entirely_nan():
    # every label unmatched for every study in the batch -- callers must skip
    # this batch rather than get a NaN loss silently
    logits = torch.tensor([[2.0, -1.0]])
    targets = torch.full((1, 2), float("nan"))

    try:
        masked_bce_loss(logits, targets)
        assert False, "expected a ValueError for an all-NaN batch"
    except ValueError:
        pass
