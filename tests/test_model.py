import pytest
import torch

from knee.model import KneeModel

# Both tests here assert output shape and pooling axis, neither of which depends
# on resolution -- and a full 16x224x224 batch OOMs a small dev box, which makes
# the suite look like it hangs. Keep the volume small enough to run anywhere.
N_SLICES, SIZE = 4, 64


class MeanStub(torch.nn.Module):
    """Backbone stub emitting each image's own mean, broadcast over the feature
    axis. An untrained efficientnet returns nearly identical features for black,
    noise and anatomy alike, so any test asserting that the model *distinguishes*
    its inputs passes with a real backbone whether the mechanism works or not."""

    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

    def forward(self, x):
        return x.mean(dim=(1, 2, 3)).unsqueeze(1).expand(-1, self.n_features)


def _stub_backbone(model):
    model.backbone = MeanStub(model.backbone.num_features)
    return model


def test_forward_pass_returns_logits_for_all_labels_no_nan():
    model = KneeModel(backbone_name="efficientnet_b0", num_labels=12, pretrained=False)
    batch = torch.randn(2, N_SLICES, 3, SIZE, SIZE)

    logits = model(batch)

    assert logits.shape == (2, 12)
    assert not torch.isnan(logits).any()


def test_mean_pools_over_slices_not_batch():
    model = KneeModel(backbone_name="efficientnet_b0", num_labels=12, pretrained=False)
    model.eval()

    # two studies with different slice content must not affect each other
    study_a = torch.randn(1, N_SLICES, 3, SIZE, SIZE)
    study_b = torch.randn(1, N_SLICES, 3, SIZE, SIZE)

    with torch.no_grad():
        logits_a_alone = model(study_a)
        logits_b_alone = model(study_b)
        logits_batched = model(torch.cat([study_a, study_b], dim=0))

    assert torch.allclose(logits_batched[0], logits_a_alone[0], atol=1e-4)
    assert torch.allclose(logits_batched[1], logits_b_alone[0], atol=1e-4)


# --- Phase 6: slot-structured input with a presence mask ------------------------


def test_forward_accepts_a_slot_structured_volume():
    # PreppedSlotDataset returns (slot, group, channel, H, W); the model folds
    # the slot and group axes together for pooling rather than the loader
    # flattening them, so the mask stays meaningful downstream
    model = KneeModel(backbone_name="efficientnet_b0", num_labels=12, pretrained=False)
    batch = torch.randn(2, 3, 2, 3, SIZE, SIZE)

    logits = model(batch)

    assert logits.shape == (2, 12)
    assert not torch.isnan(logits).any()


def test_absent_slots_do_not_drag_the_pooled_features():
    # An absent slot arrives as zeros, and an unmasked mean averages real
    # anatomy with a black square -- the more slots a study is missing, the
    # harder its features get pulled toward whatever the backbone emits for
    # black. Slot fill runs 100% (AX_FLUID) to 19% (AX_STRUCT), so that bias
    # would track missingness, which is not a property of the knee.
    #
    # The backbone is stubbed for the reason MeanStub documents.
    model = _stub_backbone(
        KneeModel(backbone_name="efficientnet_b0", num_labels=12, pretrained=False))
    model.eval()

    real = torch.full((1, 1, 2, 3, SIZE, SIZE), 4.0)
    padded = torch.cat([real, torch.zeros(1, 1, 2, 3, SIZE, SIZE)], dim=1)

    with torch.no_grad():
        alone = model(real)
        with_absent = model(padded, mask=torch.tensor([[True, False]]))
        unmasked = model(padded)

    # present-only mean is 4.0; the unmasked mean over four images is 2.0
    assert torch.allclose(with_absent, alone, atol=1e-5)
    assert not torch.allclose(unmasked, alone, atol=1e-2), "unmasked pooling must differ"


def test_a_study_with_no_present_slot_still_returns_finite_logits():
    # the submission path must never raise: a study whose every slot failed is
    # already handled by build_submission's 0.5 fallback, but the model should
    # not be the thing that explodes first
    model = KneeModel(backbone_name="efficientnet_b0", num_labels=12, pretrained=False)
    model.eval()

    with torch.no_grad():
        logits = model(torch.zeros(1, 2, 1, 3, SIZE, SIZE), mask=torch.tensor([[False, False]]))

    assert logits.shape == (1, 12)
    assert torch.isfinite(logits).all()


# --- Phase 6 c2: per-diagnosis attention over slots -----------------------------


def _attn_model():
    return KneeModel(backbone_name="efficientnet_b0", num_labels=12, pretrained=False,
                     head="slot_attention")


def test_slot_attention_returns_logits_for_all_labels():
    model = _attn_model()
    logits = model(torch.randn(2, 3, 2, 3, SIZE, SIZE),
                   mask=torch.tensor([[True, True, False], [True, False, False]]))

    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()


def test_slot_attention_gives_absent_slots_no_weight():
    # the whole reason the mask reaches the head: an absent slot must not be
    # something a diagnosis can choose to look at
    model = _attn_model()
    model.eval()
    mask = torch.tensor([[True, False, False]])

    with torch.no_grad():
        weights = model.attention_weights(torch.randn(1, 3, 2, 3, SIZE, SIZE), mask)

    assert weights.shape == (1, 3, 12)
    assert torch.allclose(weights[0, 0], torch.ones(12))
    assert float(weights[0, 1:].abs().max()) == 0.0


def test_slot_attention_lets_different_diagnoses_read_different_slots():
    # A mean-pool head cannot express "MCL lives in the coronal plane"; this one
    # has to be able to, or c2 is just c1 with more parameters. Stubbed backbone
    # so the slots carry genuinely different features -- an untrained one makes
    # every slot look alike and the softmax comes out uniform whatever the head
    # does.
    model = _stub_backbone(_attn_model())
    model.eval()

    slots = torch.stack([torch.full((2, 3, SIZE, SIZE), float(s)) for s in range(4)])
    with torch.no_grad():
        weights = model.attention_weights(slots.unsqueeze(0),
                                          torch.ones(1, 4, dtype=torch.bool))

    spread = weights[0].std(dim=0)
    assert float(spread.max()) > 0, "every diagnosis attends identically -- not per-diagnosis"
    assert not torch.allclose(weights[0, :, 0], weights[0, :, 1])


def test_slot_attention_survives_a_study_with_no_present_slot():
    model = _attn_model()
    model.eval()

    with torch.no_grad():
        logits = model(torch.zeros(1, 2, 1, 3, SIZE, SIZE),
                       mask=torch.zeros(1, 2, dtype=torch.bool))

    assert torch.isfinite(logits).all()


def test_slot_attention_requires_a_slot_axis():
    # the flat Phase 2 volume has no slot axis to attend over, and silently
    # attending over slices instead would be a different model wearing the same
    # config name
    with pytest.raises(ValueError):
        _attn_model()(torch.randn(1, 4, 3, SIZE, SIZE))


def test_backbone_kwargs_reach_timm():
    # DINOv2 is patch-14 with a default img_size of 518; built at the default it
    # rejects a 224 input on the position embedding. c3/c4 need to say what size
    # they are training at, and only timm can act on that.
    model = KneeModel(backbone_name="vit_small_patch14_dinov2.lvd142m", num_labels=12,
                      pretrained=False, head="slot_attention", img_size=SIZE * 2 - 2)

    logits = model(torch.randn(1, 2, 1, 3, SIZE * 2 - 2, SIZE * 2 - 2),
                   mask=torch.tensor([[True, False]]))

    assert logits.shape == (1, 12)
    assert model.backbone.num_features == 384
