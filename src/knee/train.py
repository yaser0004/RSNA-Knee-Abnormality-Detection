import datetime
import random
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


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE over only the non-NaN targets. Lexical/pseudo-labels leave a study's
    label NaN when no rule matches (see knee.reports.lexical_label) -- that
    means "no evidence", not "negative", so it must never enter the loss."""
    mask = ~torch.isnan(targets)
    if not mask.any():
        raise ValueError("all targets in this batch are NaN -- nothing to train on")
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask])


def make_folds(study_uids: list[str], n_folds: int = 5, seed: int = 0) -> dict[str, int]:
    """Deterministic, order-independent study-grouped fold assignment. Per the
    plan's experiment discipline this must be frozen across every experiment --
    sorting before shuffling means the result never depends on the order
    study_uids happened to come in (a different CSV read, a different pandas
    version) so a later run can never silently drift from an earlier one."""
    sorted_uids = sorted(study_uids)
    rng = random.Random(seed)
    shuffled = sorted_uids.copy()
    rng.shuffle(shuffled)
    return {uid: i % n_folds for i, uid in enumerate(shuffled)}


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
        "paired_delta": paired_delta,
        "train_minutes": train_minutes,
        "inference_seconds": inference_seconds,
        "promoted": promoted,
    }
    for label, column in _LABEL_TO_CSV_COLUMN.items():
        row[column] = per_label_auc.get(label, float("nan"))

    df = pd.DataFrame([row])[existing_header]
    df.to_csv(csv_path, mode="a", index=False, header=False)


def train_one_epoch(model, loader: DataLoader, optimizer, device: str = "cpu") -> float:
    """One training epoch. A batch where every label is NaN for every study in
    it (masked_bce_loss's ValueError) is skipped rather than crashing the run
    -- possible with a small batch size and a low-coverage lexical label."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        try:
            loss = masked_bce_loss(logits, labels)
        except ValueError:
            continue
        loss.backward()
        optimizer.step()
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
        for images, labels, _ in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_labels.append(labels.numpy())
            all_preds.append(probs.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)
