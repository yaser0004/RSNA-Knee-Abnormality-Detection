import torch

from knee.model import KneeModel


def test_forward_pass_returns_logits_for_all_labels_no_nan():
    model = KneeModel(backbone_name="efficientnet_b0", num_labels=12, pretrained=False)
    batch = torch.randn(2, 16, 3, 224, 224)

    logits = model(batch)

    assert logits.shape == (2, 12)
    assert not torch.isnan(logits).any()


def test_mean_pools_over_slices_not_batch():
    model = KneeModel(backbone_name="efficientnet_b0", num_labels=12, pretrained=False)
    model.eval()

    # two studies with different slice content must not affect each other
    study_a = torch.randn(1, 16, 3, 224, 224)
    study_b = torch.randn(1, 16, 3, 224, 224)

    with torch.no_grad():
        logits_a_alone = model(study_a)
        logits_b_alone = model(study_b)
        logits_batched = model(torch.cat([study_a, study_b], dim=0))

    assert torch.allclose(logits_batched[0], logits_a_alone[0], atol=1e-4)
    assert torch.allclose(logits_batched[1], logits_b_alone[0], atol=1e-4)
