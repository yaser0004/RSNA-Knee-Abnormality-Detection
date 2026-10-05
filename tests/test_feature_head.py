import numpy as np
import pytest
import torch

from knee.feature_head import FeatureHead, window_indices
from knee.model import KneeModel

D, S, G, B = 16, 6, 9, 3


class MeanStub(torch.nn.Module):
    """Backbone emitting each image's mean over the feature axis, so the model
    under test sees distinct, reproducible features without a real network."""

    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.num_features = n_features

    def forward(self, x):
        return x.mean(dim=(1, 2, 3)).unsqueeze(1).expand(-1, self.n_features).contiguous()


def _tokens(batch=B, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, S, G, D, generator=g)


def _full_mask(batch=B):
    return torch.ones(batch, S, dtype=torch.bool)


def test_logits_match_group_attention_head_on_the_same_features():
    # the equivalence test the earlier head changes had: a cached-feature head
    # must be the same readout as the trained model's, or its OOF is not
    # comparable to any member's
    model = KneeModel(backbone_name="efficientnet_b0", num_labels=12, pretrained=False,
                      head="group_attention")
    model.backbone = MeanStub(D)
    model.slot_scores = torch.nn.Linear(D, 12)
    model.label_readout = torch.nn.Parameter(torch.randn(12, D))
    model.label_bias = torch.nn.Parameter(torch.randn(12))
    head = FeatureHead(D)
    head.load_state_dict({"slot_scores.weight": model.slot_scores.weight.data,
                          "slot_scores.bias": model.slot_scores.bias.data,
                          "label_readout": model.label_readout.data,
                          "label_bias": model.label_bias.data})
    tokens = _tokens()
    mask = _full_mask()
    mask[1, 4:] = False

    flat = tokens.reshape(B, S * G, D)
    present = mask.repeat_interleave(G, dim=1)
    expected, _ = model._attend(flat, present)
    got = head(tokens, mask)

    assert torch.allclose(got, expected, atol=1e-6)


def test_features_of_an_absent_slot_do_not_move_the_logits():
    head = FeatureHead(D)
    tokens = _tokens()
    mask = _full_mask()
    mask[:, 3] = False
    perturbed = tokens.clone()
    perturbed[:, 3] += 100.0

    assert torch.allclose(head(tokens, mask), head(perturbed, mask), atol=1e-6)


def test_study_with_no_present_slot_is_finite_and_bias_only():
    head = FeatureHead(D)
    tokens = _tokens(batch=1)
    mask = torch.zeros(1, S, dtype=torch.bool)

    logits = head(tokens, mask)

    assert torch.isfinite(logits).all()
    assert torch.allclose(logits, head.label_bias.unsqueeze(0), atol=1e-6)


def test_window_subset_is_what_the_head_reads():
    head = FeatureHead(D)
    tokens = _tokens()
    mask = _full_mask()
    idx = torch.tensor([0, 2, 4, 6, 8])

    assert torch.allclose(head(tokens, mask, windows=idx), head(tokens[:, :, idx], mask), atol=1e-6)


def test_wrong_rank_is_rejected():
    head = FeatureHead(D)
    with pytest.raises(ValueError, match=r"\[batch, n_slots, n_groups"):
        head(torch.randn(B, S * G, D), _full_mask())


def test_eval_windows_are_the_fixed_spread_and_deterministic():
    assert window_indices(9, 5, random=False).tolist() == [0, 2, 4, 6, 8]
    assert window_indices(9, 9, random=False).tolist() == list(range(9))
    assert window_indices(9, 5, random=False).tolist() == window_indices(9, 5, random=False).tolist()


def test_train_windows_are_a_sorted_subset_of_the_stored_ones():
    gen = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        idx = window_indices(9, 5, random=True, rng=gen)
        assert len(idx) == 5 and len(set(idx.tolist())) == 5
        assert idx.tolist() == sorted(idx.tolist())
        assert idx.min() >= 0 and idx.max() <= 8
        seen.add(tuple(idx.tolist()))
    assert len(seen) > 20  # it actually varies


def test_asking_for_more_windows_than_stored_is_an_error():
    with pytest.raises(ValueError, match="stored"):
        window_indices(5, 9, random=False)
