import datetime
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Must match results/experiments.csv's header exactly -- log_experiment writes
# rows by position, not by name, so any drift here silently misaligns columns.
_LABEL_TO_CSV_COLUMN = {
    "ACL": "acl_auc",
    "MCL": "mcl_auc",
    "Medial Meniscus": "medial_meniscus_auc",
    "Lateral Meniscus": "lateral_meniscus_auc",
    "Medial OA": "medial_oa_auc",
    "Lateral OA": "lateral_oa_auc",
    "PF OA": "pf_oa_auc",
    "Effusion": "effusion_auc",
    "Synovitis": "synovitis_auc",
    "Baker's": "bakers_auc",
    "Contusion": "contusion_auc",
    "Fracture": "fracture_auc",
}


def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """BCE over only the non-NaN targets. Lexical/pseudo-labels leave a study's
    label NaN when no rule matches (see knee.reports.lexical_label) -- that
    means "no evidence", not "negative", so it must never enter the loss.

    `weights` is an optional per-element sample weight, same shape as targets,
    for discounting rows the label sources disagree about (see
    knee.reports.confidence_from_agreement). A soft target cannot express that:
    a disputed row sits at an irreducible loss floor and teaches the model to
    answer 0.5, and only a weight can make it count for less.

    Combined as a weighted *mean*, not a weighted sum. A weighted sum's scale
    moves with the weights, so a down-weighted run would also be a run at a lower
    effective learning rate and the screen could not tell the two apart."""
    mask = ~torch.isnan(targets)
    if not mask.any():
        raise ValueError("all targets in this batch are NaN -- nothing to train on")
    if weights is None:
        return F.binary_cross_entropy_with_logits(logits[mask], targets[mask])

    w = weights[mask]
    total = w.sum()
    if total <= 0:
        raise ValueError("every sample weight in this batch is zero -- nothing to "
                         "train on; a weight floor should keep disputed rows alive")
    per_element = F.binary_cross_entropy_with_logits(
        logits[mask], targets[mask], reduction="none")
    return (per_element * w).sum() / total


def make_folds(
    study_uids: list[str],
    n_folds: int = 5,
    seed: int = 0,
    groups: dict[str, str] | None = None,
) -> dict[str, int]:
    """Deterministic, order-independent study-grouped fold assignment. Per the
    plan's experiment discipline this must be frozen across every experiment --
    sorting before shuffling means the result never depends on the order
    study_uids happened to come in (a different CSV read, a different pandas
    version) so a later run can never silently drift from an earlier one.

    `groups` maps a study to a key that must not straddle folds -- the report
    text hash, in practice. 49 byte-identical report groups cover 183 studies
    and 45 of them straddled folds_primary_v2, so ~4% of that OOF was scored
    against a report the model had already trained on. A study missing from
    `groups` is its own group.

    Groups are assigned largest-first to whichever fold is currently smallest,
    rather than round-robin: one real group has 37 members, and round-robin
    over groups would drop that whole block into one fold. Passing groups=None
    keeps the original path byte-for-byte, because folds_primary_v2 is frozen
    and every Phase 4/5 number is paired against it."""
    sorted_uids = sorted(study_uids)
    rng = random.Random(seed)
    shuffled = sorted_uids.copy()
    rng.shuffle(shuffled)

    if groups is None:
        return {uid: i % n_folds for i, uid in enumerate(shuffled)}

    members: dict[str, list[str]] = {}
    for uid in shuffled:
        members.setdefault(groups.get(uid, uid), []).append(uid)

    # shuffled order breaks size ties, so the result depends on the seed rather
    # than on how group keys happen to sort
    order = sorted(members, key=lambda key: -len(members[key]))
    sizes = [0] * n_folds
    folds: dict[str, int] = {}
    for key in order:
        fold = min(range(n_folds), key=lambda f: (sizes[f], f))
        sizes[fold] += len(members[key])
        for uid in members[key]:
            folds[uid] = fold
    return folds


class Timer:
    """Accumulating wall-clock stopwatch for experiment logging. Each with-block
    adds to the running total, so timing N epochs is N enter/exit pairs and
    train_minutes comes straight off .minutes at log_experiment time. The Phase 1
    run logged 0.0 into that column for every fold because this timing was
    hand-rolled in the notebook and never wired up (see NOTES.md 2026-08-08)."""

    def __init__(self):
        self.total_seconds = 0.0
        self._start: float | None = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self.total_seconds += time.perf_counter() - self._start
        self._start = None
        return False

    @property
    def minutes(self) -> float:
        return self.total_seconds / 60.0


def load_gold_holdout(csv_path: str | Path) -> frozenset[str]:
    """The studies carrying gold rubric labels (results/gold_study_uids.csv, one
    StudyInstanceUID per row). Phase 4+ must never train on these -- Phase 1 did,
    silently, which is why its gold-LOO gate had to be substituted with the LB
    back-out (see NOTES.md 2026-08-08). Loading them as one explicit set makes the
    exclusion a single greppable call instead of a pandas filter each notebook
    re-derives with its own chance to get it wrong."""
    uids = pd.read_csv(csv_path)["StudyInstanceUID"].astype(str)
    if len(uids) == 0:
        raise ValueError(f"{csv_path} lists no gold studies -- an empty holdout would "
                         "silently disable the exclusion")
    return frozenset(uids)


def train_val_split(
    folds: dict[str, int],
    val_fold: int,
    exclude_uids: frozenset[str] | set[str] = frozenset(),
) -> tuple[list[str], list[str]]:
    """Train/val UID lists for one fold over the frozen assignment. Excluded
    studies (the gold holdout) are dropped from BOTH lists: never trained on, and
    not silently mixed into an OOF score whose job is to measure pseudo-label
    transfer on non-gold studies -- the caller scores them separately against
    their real rubric labels."""
    train, val = [], []
    for uid, fold in sorted(folds.items()):
        if uid in exclude_uids:
            continue
        (val if fold == val_fold else train).append(uid)
    return train, val


def log_experiment(
    csv_path: str | Path,
    *,
    git_sha: str,
    config_hash: str,
    hypothesis: str,
    fold_set: str,
    seed: int,
    per_label_auc: dict[str, float],
    macro_auc: float,
    paired_delta: float,
    train_minutes: float,
    inference_seconds: float,
    promoted: bool,
    gold_macro: float = float("nan"),
) -> None:
    """Append one row to results/experiments.csv. Per the plan's experiment
    discipline this is the only way a run should ever be recorded -- "if a run
    isn't in here it didn't happen." Column order is read from the existing
    file's own header rather than assumed, so this can never silently drift
    out of sync with results/experiments.csv."""
    csv_path = Path(csv_path)
    existing_header = csv_path.read_text().splitlines()[0].split(",")

    row = {
        "date": datetime.date.today().isoformat(),
        "git_sha": git_sha,
        "config_hash": config_hash,
        "hypothesis": hypothesis,
        "fold_set": fold_set,
        "seed": seed,
        "macro_auc": macro_auc,
        # the second validation tier: the 58 rubric-graded studies no model trains
        # on. Promotion needs it and it lived only in NOTES prose until 2026-09-07.
        # Defaults to NaN rather than 0 -- a run that never scored gold is not a
        # run that scored zero. Dropped silently on a CSV whose header predates it.
        "gold_macro": gold_macro,
        "paired_delta": paired_delta,
        "train_minutes": train_minutes,
        "inference_seconds": inference_seconds,
        "promoted": promoted,
    }
    for label, column in _LABEL_TO_CSV_COLUMN.items():
        row[column] = per_label_auc.get(label, float("nan"))

    df = pd.DataFrame([row])[existing_header]
    df.to_csv(csv_path, mode="a", index=False, header=False)


def differential_param_groups(model, backbone_lr: float, head_lr: float) -> list[dict]:
    """Optimizer param groups running the pretrained backbone slower than the
    freshly-initialised head.

    A self-supervised or ImageNet backbone already represents images well; the
    head is random. A single learning rate has to be either low enough not to
    wreck the backbone or high enough to train the head, and it cannot be both.
    Splitting them is what makes a large pretrained trunk usable on 3,486
    studies at all.

    Everything not under `backbone.` goes with the head -- that includes the
    slot-attention parameters, which are as freshly initialised as the linear
    head and would be effectively untrained at the backbone's rate.

    ponytail: no explicit block freezing. A backbone at 8e-6 against a head at
    1e-3 is already a near-freeze, and "last N blocks" is spelled differently on
    every architecture. Add it only if a screen shows the trunk drifting.
    """
    backbone, head = [], []
    for name, param in model.named_parameters():
        (backbone if name.startswith("backbone.") else head).append(param)
    return [{"params": backbone, "lr": backbone_lr},
            {"params": head, "lr": head_lr}]


def _unpack_batch(batch):
    """Read a batch from any loader shape: PreppedSlotDataset yields
    (image, mask, labels, uid), or (image, mask, labels, weights, uid) when
    sample weights are configured, and the Phase 2 PreppedStudyDataset yields
    (image, labels, uid). Returns (images, mask, labels, weights) with None for
    whatever the shape does not carry, so one training loop serves all three
    while prep v1 is still the artifact behind a scored submission."""
    if len(batch) == 5:
        images, mask, labels, weights, _ = batch
        return images, mask, labels, weights
    if len(batch) == 4:
        images, mask, labels, _ = batch
        return images, mask, labels, None
    images, labels, _ = batch
    return images, None, labels, None


def train_one_epoch(model, loader: DataLoader, optimizer, device: str = "cpu",
                    scaler=None, scheduler=None) -> float:
    """One training epoch. A batch where every label is NaN for every study in
    it (masked_bce_loss's ValueError) is skipped rather than crashing the run
    -- possible with a small batch size and a low-coverage lexical label.

    `scaler` is a torch.amp.GradScaler; passing one runs the forward under
    autocast on the matching device type. Left None the arithmetic is exactly
    what Phase 4 ran, so every pre-AMP result stays reproducible from this
    function rather than from a copy of it.

    `scheduler` is stepped once per optimizer step, not once per epoch: OneCycle
    and every other per-batch schedule define their cycle in steps, and stepping
    per epoch would traverse 1/len(loader) of the cycle and leave the learning
    rate stranded near its floor.

    A skipped all-NaN batch steps neither the optimizer nor the scheduler --
    there was no update to schedule."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"

    for batch in loader:
        images, mask, labels, weights = _unpack_batch(batch)
        images = images.to(device)
        labels = labels.to(device)
        mask = mask.to(device) if mask is not None else None
        weights = weights.to(device) if weights is not None else None
        optimizer.zero_grad()

        if scaler is None:
            logits = model(images, mask=mask)
            try:
                loss = masked_bce_loss(logits, labels, weights)
            except ValueError:
                continue
            loss.backward()
            optimizer.step()
        else:
            with torch.autocast(autocast_device, enabled=scaler.is_enabled()):
                logits = model(images, mask=mask)
                try:
                    loss = masked_bce_loss(logits, labels, weights)
                except ValueError:
                    continue
            # BCE-with-logits under fp16 can underflow the backward pass; the
            # scaler is what keeps those gradients representable.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        n_batches += 1

    if n_batches == 0:
        raise ValueError("no batch in this epoch had a non-NaN label")
    return total_loss / n_batches


def evaluate(model, loader: DataLoader, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a loader, returning (y_true, y_pred) as full
    [n, num_labels] arrays. NaN in y_true is preserved, not dropped --
    knee.metrics.per_label_auc masks it per column when scoring."""
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            images, mask, labels, _ = _unpack_batch(batch)
            images = images.to(device)
            mask = mask.to(device) if mask is not None else None
            logits = model(images, mask=mask)
            probs = torch.sigmoid(logits)
            all_labels.append(labels.numpy())
            all_preds.append(probs.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)
