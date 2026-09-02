import torch

from knee.model import KneeModel

# Both tests here assert output shape and pooling axis, neither of which depends
# on resolution -- and a full 16x224x224 batch OOMs a small dev box, which makes
# the suite look like it hangs. Keep the volume small enough to run anywhere.
N_SLICES, SIZE = 4, 64


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
