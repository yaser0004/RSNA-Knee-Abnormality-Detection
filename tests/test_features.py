import numpy as np
import pytest
import torch

from knee.features import extract_study_features, load_feature_cache, save_feature_cache

D, S, G = 8, 6, 9


class CountingStub(torch.nn.Module):
    """Emits each image's mean over D features and records how many images it saw,
    so a test can assert that absent slots were never run through the backbone."""

    def __init__(self):
        super().__init__()
        self.seen = 0
        self.num_features = D  # as a timm backbone has

    def forward(self, x):
        self.seen += x.shape[0]
        return x.mean(dim=(1, 2, 3)).unsqueeze(1).expand(-1, D).contiguous()


def _volume(present):
    g = torch.Generator().manual_seed(1)
    vol = torch.rand(S, G, 3, 12, 12, generator=g)
    vol[~present] = 0
    return vol


def test_absent_slots_get_zero_features_and_are_never_run_through_the_backbone():
    present = torch.tensor([True, True, False, True, False, False])
    backbone = CountingStub()

    feats = extract_study_features(backbone, _volume(present), present)

    assert feats.shape == (S, G, D)
    assert backbone.seen == int(present.sum()) * G
    assert torch.count_nonzero(feats[~present]) == 0
    assert torch.count_nonzero(feats[present]) > 0


def test_study_with_no_present_slot_yields_all_zero_features():
    present = torch.zeros(S, dtype=torch.bool)
    backbone = CountingStub()

    feats = extract_study_features(backbone, _volume(present), present)

    assert backbone.seen == 0
    assert torch.count_nonzero(feats) == 0


def test_wrong_volume_shape_is_rejected():
    with pytest.raises(ValueError, match="n_slots, n_groups"):
        extract_study_features(CountingStub(), torch.zeros(S, 3, 12, 12), torch.ones(S, dtype=torch.bool))


def _cache(n=4, seed=0):
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n, S, G, D)).astype(np.float16)
    mask = rng.random((n, S)) > 0.3
    feats[~mask] = 0
    uids = [f"1.2.3.{i}" for i in range(n)]
    return feats, mask, uids


def test_cache_round_trips_and_stays_half_precision(tmp_path):
    feats, mask, uids = _cache()
    path = tmp_path / "feat.npz"

    save_feature_cache(path, feats, mask, uids, backbone="stub")
    got = load_feature_cache(path)

    assert got.features.dtype == np.float16
    assert np.array_equal(got.features, feats)
    assert np.array_equal(got.mask, mask)
    assert list(got.uids) == uids
    assert got.backbone == "stub"


def test_loading_against_a_different_uid_order_is_refused(tmp_path):
    feats, mask, uids = _cache()
    path = tmp_path / "feat.npz"
    save_feature_cache(path, feats, mask, uids, backbone="stub")

    with pytest.raises(ValueError, match="order"):
        load_feature_cache(path, expect_uids=list(reversed(uids)))


def test_loading_with_the_expected_order_succeeds(tmp_path):
    feats, mask, uids = _cache()
    path = tmp_path / "feat.npz"
    save_feature_cache(path, feats, mask, uids, backbone="stub")

    assert list(load_feature_cache(path, expect_uids=uids).uids) == uids


def test_saving_features_for_an_absent_slot_is_refused(tmp_path):
    feats, mask, uids = _cache()
    feats[~mask] = 1.0  # a nonzero feature under a False mask would leak into any unmasked mean

    with pytest.raises(ValueError, match="absent"):
        save_feature_cache(tmp_path / "feat.npz", feats, mask, uids, backbone="stub")


def test_row_count_must_match_the_uids(tmp_path):
    feats, mask, uids = _cache()
    with pytest.raises(ValueError, match="rows"):
        save_feature_cache(tmp_path / "feat.npz", feats, mask, uids[:-1], backbone="stub")
