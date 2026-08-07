from typing import Callable

import numpy as np
import pandas as pd

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
