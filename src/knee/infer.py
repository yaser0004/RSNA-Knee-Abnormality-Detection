from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

LABEL_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def build_submission(
    study_uids: list[str], predict_fn: Callable[[str], np.ndarray]
) -> pd.DataFrame:
    """Run predict_fn per study, falling back to 0.5 for every label if a
    study's DICOMs fail to decode or prediction otherwise raises. A submission
    that errors on the hidden test set loses the competition outright, so this
    fallback is never optional."""
    rows = []
    for uid in study_uids:
        try:
            preds = predict_fn(uid)
        except Exception:
            preds = np.full(len(LABEL_COLUMNS), 0.5)
        rows.append([uid, *preds])
    return pd.DataFrame(rows, columns=["StudyInstanceUID"] + LABEL_COLUMNS)


def rank_mean(per_model: list[np.ndarray]) -> np.ndarray:
    """Combine fold/model predictions by averaging their per-label ranks across
    studies, rather than averaging probabilities.

    Macro AUC is invariant under any increasing map of a label's scores, so it
    reads order and nothing else. A probability mean therefore lets the member
    with the widest spread dominate the combination -- an overconfident model
    that is wrong outvotes two diffident models that are right. Averaging ranks
    gives each member one vote in exactly the currency the metric reads.

    **Measured 2026-09-07, and it does not beat a probability mean on our own
    members: prob-mean won 11 of 11 combinations of the Phase 6 fold-0 arms,
    paired +0.0018 [+0.0009, +0.0027].** The argument above assumes members on
    incomparable scales, which is the case it protects against; our arms are
    sigmoid outputs of similarly-trained models on one target, so they are
    already comparable, and ranking throws away agreement strength that is real
    signal. Keep this for blending members that genuinely differ in calibration
    -- a public model's outputs against ours -- and use a probability mean for a
    homogeneous fold/config ensemble.

    Each member is (n_studies, n_labels) and every member must cover the same
    studies in the same order. Ties keep equal ranks (an ordinal ranking would
    invent an ordering the models never expressed). The result is rescaled to
    [0, 1] because a submission has to be, not because it carries probability
    meaning any more -- it does not, and nothing downstream should read it as
    calibrated."""
    if not per_model:
        raise ValueError("rank_mean needs at least one member")
    shapes = {arr.shape for arr in per_model}
    if len(shapes) != 1:
        raise ValueError(f"members must share a shape, got {sorted(shapes)}")

    ranked = np.mean([rankdata(arr, axis=0) for arr in per_model], axis=0)
    n_studies = ranked.shape[0]
    if n_studies == 1:
        return np.full_like(ranked, 0.5)
    return (ranked - 1) / (n_studies - 1)
