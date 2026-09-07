import collections
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from knee.infer import LABEL_COLUMNS
from knee.model import KneeModel
from knee.train import (
    Timer,
    differential_param_groups,
    evaluate,
    load_gold_holdout,
    log_experiment,
    make_folds,
    masked_bce_loss,
    train_one_epoch,
    train_val_split,
)

_REAL_EXPERIMENTS_CSV = Path(__file__).resolve().parents[1] / "results" / "experiments.csv"
_REAL_GOLD_CSV = Path(__file__).resolve().parents[1] / "results" / "gold_study_uids.csv"


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


def test_timer_accumulates_across_blocks_and_reports_minutes():
    import time

    timer = Timer()
    with timer:
        time.sleep(0.02)
    first = timer.total_seconds
    with timer:
        time.sleep(0.02)
    second_block = timer.total_seconds - first

    assert 0.01 < first < 1.0
    assert 0.01 < second_block < 1.0
    assert abs(timer.minutes - timer.total_seconds / 60.0) < 1e-12


def test_load_gold_holdout_reads_the_real_frozen_file():
    holdout = load_gold_holdout(_REAL_GOLD_CSV)

    assert len(holdout) == 58
    assert all(isinstance(u, str) and u.startswith("1.2.826.") for u in holdout)


def test_load_gold_holdout_rejects_an_empty_file(tmp_path):
    csv_path = tmp_path / "gold.csv"
    pd.DataFrame({"StudyInstanceUID": []}).to_csv(csv_path, index=False)

    try:
        load_gold_holdout(csv_path)
        assert False, "expected ValueError for an empty gold file"
    except ValueError:
        pass


def test_load_gold_holdout_keeps_uids_as_strings_even_if_numeric_looking(tmp_path):
    # a UID that looks numeric must not become an int and silently stop matching
    # the string keys in the folds dict -- that would disable the exclusion
    csv_path = tmp_path / "gold.csv"
    pd.DataFrame({"StudyInstanceUID": ["12345", "1.2.826.x"]}).to_csv(csv_path, index=False)

    assert load_gold_holdout(csv_path) == frozenset({"12345", "1.2.826.x"})


def test_train_val_split_partitions_every_non_excluded_study_exactly_once():
    folds = make_folds([f"study{i}" for i in range(50)], n_folds=5, seed=0)
    exclude = frozenset({"study0", "study7"})

    train, val = train_val_split(folds, val_fold=2, exclude_uids=exclude)

    assert set(train) | set(val) == set(folds) - exclude
    assert not set(train) & set(val)
    assert all(folds[u] != 2 for u in train)
    assert all(folds[u] == 2 for u in val)


def test_train_val_split_drops_excluded_studies_from_both_sides():
    folds = {"a": 0, "b": 0, "c": 1, "d": 1}

    train, val = train_val_split(folds, val_fold=1, exclude_uids=frozenset({"c"}))

    assert set(train) == {"a", "b"}
    assert set(val) == {"d"}
    assert "c" not in train and "c" not in val


def test_train_val_split_is_deterministic():
    folds = make_folds([f"study{i}" for i in range(50)], n_folds=5, seed=0)

    assert (train_val_split(folds, 3) == train_val_split(folds, 3))


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


def test_train_one_epoch_steps_the_scheduler_once_per_batch():
    """OneCycle is a per-batch schedule -- stepping it per epoch instead would
    walk 1/N of the way through the cycle and leave the LR near its floor."""
    torch.manual_seed(0)
    model = KneeModel(backbone_name="efficientnet_b0", num_labels=len(LABEL_COLUMNS), pretrained=False)
    loader = torch.utils.data.DataLoader(_TinySyntheticDataset(), batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    calls = []

    class _RecordingScheduler:
        def step(self):
            calls.append(1)

    train_one_epoch(model, loader, optimizer, device="cpu", scheduler=_RecordingScheduler())

    assert len(calls) == len(loader)


def test_train_one_epoch_accepts_a_scaler_and_still_updates_weights():
    """The AMP path with the scaler disabled must be numerically the ordinary
    path, so a CPU box runs the same code the T4 runs."""
    torch.manual_seed(0)
    model = KneeModel(backbone_name="efficientnet_b0", num_labels=len(LABEL_COLUMNS), pretrained=False)
    loader = torch.utils.data.DataLoader(_TinySyntheticDataset(), batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    before = model.head.weight.detach().clone()
    loss = train_one_epoch(model, loader, optimizer, device="cpu", scaler=scaler)
    after = model.head.weight.detach().clone()

    assert torch.isfinite(torch.tensor(loss))
    assert not torch.allclose(before, after)


def test_train_one_epoch_scaler_path_matches_the_plain_path():
    def run(use_scaler):
        torch.manual_seed(0)
        model = KneeModel(backbone_name="efficientnet_b0", num_labels=len(LABEL_COLUMNS), pretrained=False)
        loader = torch.utils.data.DataLoader(_TinySyntheticDataset(), batch_size=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=False) if use_scaler else None
        return train_one_epoch(model, loader, optimizer, device="cpu", scaler=scaler)

    assert run(True) == pytest.approx(run(False), rel=1e-6)


# --- Phase 6 Step 2: report-hash grouped folds ---------------------------------


def test_make_folds_without_groups_is_unchanged():
    # folds_primary_v2.csv is frozen and every Phase 4/5 number is paired
    # against it -- adding the group argument must not move the ungrouped path.
    uids = [f"study-{i:03d}" for i in range(50)]

    assert make_folds(uids) == make_folds(uids, groups=None)
    assert sorted(make_folds(uids).values()) == sorted(i % 5 for i in range(50))


def test_make_folds_keeps_every_member_of_a_group_in_one_fold():
    # 49 byte-identical report groups cover 183 studies; 45 of them straddled
    # folds in v2, so ~4% of the OOF was scored against a report the model had
    # already been trained on.
    uids = [f"study-{i:03d}" for i in range(60)]
    groups = {uid: f"report-{i // 4}" for i, uid in enumerate(uids)}

    folds = make_folds(uids, groups=groups)

    by_group = {}
    for uid, fold in folds.items():
        by_group.setdefault(groups[uid], set()).add(fold)
    assert all(len(f) == 1 for f in by_group.values())


def test_make_folds_stays_balanced_when_group_sizes_differ_wildly():
    # one real group has 37 members; round-robin over groups would put that
    # whole block in one fold and leave the folds visibly uneven
    uids = [f"study-{i:03d}" for i in range(100)]
    groups = {uid: ("big" if i < 37 else f"solo-{i}") for i, uid in enumerate(uids)}

    counts = collections.Counter(make_folds(uids, groups=groups).values())

    assert len(counts) == 5
    assert max(counts.values()) - min(counts.values()) <= 37


def test_make_folds_grouped_is_deterministic_and_order_independent():
    uids = [f"study-{i:03d}" for i in range(40)]
    groups = {uid: f"report-{i // 3}" for i, uid in enumerate(uids)}

    assert make_folds(uids, groups=groups) == make_folds(list(reversed(uids)), groups=groups)


def test_make_folds_treats_an_ungrouped_study_as_its_own_group():
    uids = [f"study-{i:03d}" for i in range(20)]
    groups = {"study-000": "shared", "study-001": "shared"}

    folds = make_folds(uids, groups=groups)

    assert len(folds) == 20
    assert folds["study-000"] == folds["study-001"]


def test_training_and_eval_accept_a_slot_loader_with_a_presence_mask():
    # PreppedSlotDataset yields (image, mask, labels, uid); the Phase 2 dataset
    # yields (image, labels, uid). One loop has to read both while prep v1 is
    # still the artifact behind the last scored submission.
    class SlotBatches:
        def __init__(self, n=2):
            self.n = n

        def __iter__(self):
            for _ in range(self.n):
                yield (torch.randn(2, 2, 1, 3, 32, 32),
                       torch.tensor([[True, False], [True, True]]),
                       torch.rand(2, 12).round(),
                       ["a", "b"])

        def __len__(self):
            return self.n

    model = KneeModel(backbone_name="efficientnet_b0", num_labels=12, pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss = train_one_epoch(model, SlotBatches(), optimizer)
    y_true, y_pred = evaluate(model, SlotBatches())

    assert loss == loss  # not NaN
    assert y_true.shape == (4, 12) and y_pred.shape == (4, 12)


def test_training_still_accepts_the_three_tuple_loader():
    class FlatBatches:
        def __iter__(self):
            yield torch.randn(2, 4, 3, 32, 32), torch.rand(2, 12).round(), ["a", "b"]

        def __len__(self):
            return 1

    model = KneeModel(backbone_name="efficientnet_b0", num_labels=12, pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    assert train_one_epoch(model, FlatBatches(), optimizer) == pytest.approx(
        train_one_epoch(model, FlatBatches(), optimizer), abs=1.0)


def test_differential_param_groups_splits_backbone_from_head():
    # A self-supervised backbone arrives already good at representing knees; the
    # head starts random. One learning rate for both either wrecks the backbone
    # or starves the head, which is why pilkwang runs 8e-6 against 1e-3.
    model = KneeModel(backbone_name="efficientnet_b0", num_labels=12, pretrained=False)

    groups = differential_param_groups(model, backbone_lr=8e-6, head_lr=1e-3)

    assert [g["lr"] for g in groups] == [8e-6, 1e-3]
    assert all(g["params"] for g in groups), "neither group may be empty"
    n_grouped = sum(len(g["params"]) for g in groups)
    assert n_grouped == len(list(model.parameters())), "every parameter must land in a group"


def test_differential_param_groups_puts_the_attention_head_with_the_head():
    # the c2 head's parameters are new and random, exactly like the linear head:
    # training them at the backbone's rate would leave them essentially untrained
    model = KneeModel(backbone_name="efficientnet_b0", num_labels=12, pretrained=False,
                      head="slot_attention")

    backbone_group, head_group = differential_param_groups(model, 8e-6, 1e-3)

    backbone_ids = {id(p) for p in backbone_group["params"]}
    for name, param in model.named_parameters():
        if not name.startswith("backbone."):
            assert id(param) not in backbone_ids, f"{name} belongs with the head"


def test_log_experiment_records_gold_macro_and_defaults_it_to_nan(tmp_path):
    """Promotion turns on "gold transfer not regressed", so the gold macro has to
    live in the log next to the val macro it is weighed against. Every screen this
    project has run reported gold as its second tier and none of it is in the CSV.
    Old callers that omit it must still work -- the column reads NaN, not zero,
    because a run that never scored gold is not a run that scored zero on it."""
    csv_path = tmp_path / "experiments.csv"
    csv_path.write_text(_REAL_EXPERIMENTS_CSV.read_text().splitlines()[0] + "\n")

    common = dict(
        git_sha="abc1234",
        config_hash="c2_6slot_attn",
        hypothesis="h",
        fold_set="primary_v3_fold0",
        seed=0,
        per_label_auc={"Effusion": 0.7},
        macro_auc=0.8607,
        paired_delta=0.0,
        train_minutes=51.0,
        inference_seconds=float("nan"),
        promoted=False,
    )
    log_experiment(csv_path, gold_macro=0.8759, **common)
    log_experiment(csv_path, **common)

    df = pd.read_csv(csv_path)
    assert df.loc[0, "gold_macro"] == 0.8759
    assert pd.isna(df.loc[1, "gold_macro"])


class TestWeightedMaskedBceLoss:
    """Per-row sample weights on the BCE, for the confidence-weighting screen.

    A weighted mean (sum(w*bce)/sum(w)), not a weighted sum: the weighted sum's
    scale moves with the weights, so a down-weighted run would also be a
    lower-effective-learning-rate run and the screen would confound the two.
    """

    def _fixture(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 3)
        targets = torch.tensor([[1.0, 0.0, float("nan")],
                                [0.0, 1.0, 1.0],
                                [1.0, float("nan"), 0.0],
                                [0.0, 0.0, 1.0]])
        return logits, targets

    def test_no_weights_is_unchanged(self):
        logits, targets = self._fixture()
        assert masked_bce_loss(logits, targets, weights=None) == masked_bce_loss(logits, targets)

    def test_uniform_weights_match_the_unweighted_loss(self):
        logits, targets = self._fixture()
        for value in (1.0, 0.5, 7.0):
            weighted = masked_bce_loss(logits, targets, weights=torch.full_like(targets, value))
            assert torch.allclose(weighted, masked_bce_loss(logits, targets), atol=1e-6)

    def test_a_zero_weight_removes_that_element_entirely(self):
        """Equal to the loss computed on targets where the element is NaN --
        which is the definition of "this row contributes nothing"."""
        logits, targets = self._fixture()
        w = torch.ones_like(targets)
        w[1, 2] = 0.0
        dropped = targets.clone()
        dropped[1, 2] = float("nan")
        assert torch.allclose(masked_bce_loss(logits, targets, weights=w),
                              masked_bce_loss(logits, dropped), atol=1e-6)

    def test_weight_shifts_the_loss_toward_the_upweighted_element(self):
        logits, targets = self._fixture()
        per_element = F.binary_cross_entropy_with_logits(
            logits, torch.nan_to_num(targets), reduction="none")
        hardest = (per_element * ~torch.isnan(targets)).argmax()
        w = torch.ones_like(targets)
        w.view(-1)[hardest] = 10.0
        assert masked_bce_loss(logits, targets, weights=w) > masked_bce_loss(logits, targets)

    def test_weights_on_nan_targets_are_ignored(self):
        logits, targets = self._fixture()
        a = torch.ones_like(targets)
        b = a.clone()
        b[torch.isnan(targets)] = 99.0
        assert torch.allclose(masked_bce_loss(logits, targets, weights=a),
                              masked_bce_loss(logits, targets, weights=b), atol=1e-6)

    def test_all_zero_weights_raise_rather_than_return_nan(self):
        logits, targets = self._fixture()
        with pytest.raises(ValueError, match="weight"):
            masked_bce_loss(logits, targets, weights=torch.zeros_like(targets))
