from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from knee.infer import LABEL_COLUMNS
from knee.model import KneeModel
from knee.train import evaluate, log_experiment, make_folds, masked_bce_loss, train_one_epoch

_REAL_EXPERIMENTS_CSV = Path(__file__).resolve().parents[1] / "results" / "experiments.csv"


class _TinySyntheticDataset(Dataset):
    """4 studies, 2 slices each at 32x32 -- big enough to exercise the real
    KneeModel forward/backward path, small enough to run in a unit test.
    Label column 0 is always non-NaN so no batch can be all-NaN."""

    def __init__(self, n_studies: int = 4, n_slices: int = 2, size: int = 64):
        self.n_studies = n_studies
        self.n_slices = n_slices
        self.size = size

    def __len__(self):
        return self.n_studies

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        image = torch.rand(self.n_slices, 3, self.size, self.size)
        labels = torch.full((len(LABEL_COLUMNS),), float("nan"))
        labels[0] = float(idx % 2)
        return image, labels, f"study{idx}"


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


def test_make_folds_assigns_every_study_exactly_once():
    study_uids = [f"study{i}" for i in range(23)]

    folds = make_folds(study_uids, n_folds=5, seed=0)

    assert set(folds.keys()) == set(study_uids)
    assert set(folds.values()) == {0, 1, 2, 3, 4}


def test_make_folds_is_deterministic_across_calls():
    study_uids = [f"study{i}" for i in range(50)]

    folds_a = make_folds(study_uids, n_folds=5, seed=0)
    folds_b = make_folds(study_uids, n_folds=5, seed=0)

    assert folds_a == folds_b


def test_make_folds_is_independent_of_input_order():
    # the plan requires the fold assignment to be frozen across all
    # experiments -- if a later pandas read reorders StudyInstanceUID, the
    # assignment must not silently change underneath already-logged runs
    study_uids = [f"study{i}" for i in range(50)]
    shuffled = list(reversed(study_uids))

    folds_original_order = make_folds(study_uids, n_folds=5, seed=0)
    folds_shuffled_order = make_folds(shuffled, n_folds=5, seed=0)

    assert folds_original_order == folds_shuffled_order


def test_make_folds_produces_roughly_balanced_folds():
    study_uids = [f"study{i}" for i in range(100)]

    folds = make_folds(study_uids, n_folds=5, seed=0)

    counts = [list(folds.values()).count(i) for i in range(5)]
    assert max(counts) - min(counts) <= 1


def test_log_experiment_appends_a_row_matching_the_real_csv_header(tmp_path):
    csv_path = tmp_path / "experiments.csv"
    csv_path.write_text(_REAL_EXPERIMENTS_CSV.read_text().splitlines()[0] + "\n")

    log_experiment(
        csv_path,
        git_sha="abc1234",
        config_hash="deadbeef",
        hypothesis="test hypothesis",
        fold_set="primary_v1",
        seed=0,
        per_label_auc={"Effusion": 0.7, "ACL": 0.6},
        macro_auc=0.65,
        paired_delta=0.02,
        train_minutes=12.5,
        inference_seconds=30.0,
        promoted=True,
    )

    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert list(df.columns) == _REAL_EXPERIMENTS_CSV.read_text().splitlines()[0].split(",")
    assert df.loc[0, "effusion_auc"] == 0.7
    assert df.loc[0, "acl_auc"] == 0.6
    assert pd.isna(df.loc[0, "mcl_auc"])
    assert df.loc[0, "macro_auc"] == 0.65
    assert df.loc[0, "hypothesis"] == "test hypothesis"
    assert df.loc[0, "promoted"] == True


def test_log_experiment_appends_without_duplicating_header_on_second_call(tmp_path):
    csv_path = tmp_path / "experiments.csv"
    csv_path.write_text(_REAL_EXPERIMENTS_CSV.read_text().splitlines()[0] + "\n")

    for i in range(2):
        log_experiment(
            csv_path,
            git_sha=f"sha{i}",
            config_hash="hash",
            hypothesis="h",
            fold_set="primary_v1",
            seed=0,
            per_label_auc={"Effusion": 0.7},
            macro_auc=0.65,
            paired_delta=0.0,
            train_minutes=1.0,
            inference_seconds=1.0,
            promoted=False,
        )

    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert list(df["git_sha"]) == ["sha0", "sha1"]


def test_train_one_epoch_returns_finite_loss_and_updates_weights():
    torch.manual_seed(0)
    model = KneeModel(backbone_name="efficientnet_b0", num_labels=len(LABEL_COLUMNS), pretrained=False)
    loader = torch.utils.data.DataLoader(_TinySyntheticDataset(), batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    before = model.head.weight.detach().clone()
    loss = train_one_epoch(model, loader, optimizer, device="cpu")
    after = model.head.weight.detach().clone()

    assert torch.isfinite(torch.tensor(loss))
    assert not torch.allclose(before, after)


def test_evaluate_returns_correctly_shaped_arrays():
    torch.manual_seed(0)
    model = KneeModel(backbone_name="efficientnet_b0", num_labels=len(LABEL_COLUMNS), pretrained=False)
    loader = torch.utils.data.DataLoader(_TinySyntheticDataset(n_studies=4), batch_size=2)

    y_true, y_pred = evaluate(model, loader, device="cpu")

    assert y_true.shape == (4, len(LABEL_COLUMNS))
    assert y_pred.shape == (4, len(LABEL_COLUMNS))
    assert (y_pred >= 0).all() and (y_pred <= 1).all()
    # only label column 0 was ever populated in the synthetic dataset
    assert not any(torch.isnan(torch.tensor(y_true[:, 0])))
