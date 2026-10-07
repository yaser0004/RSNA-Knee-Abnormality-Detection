"""Report relabelling by a hosted LLM (Claude), allowed by the host in discussion thread 733965.

The trust boundary runs both ways. Outgoing, a chunk carries the study uid and the report text,
never a label column, so the 58 gold labels cannot leak into the labeller. Incoming, the model's
output is untrusted text: a row that is not valid JSON or has no uid is dropped, and any value
that is not a probability in [0, 1] becomes NaN. NaN is masked by `masked_bce_loss`; a zero
would train the model on a negative nobody asserted.
"""
import json
import math
from pathlib import Path

import pandas as pd

from knee.infer import LABEL_COLUMNS

# Findings the rubric does not score but that a report states and an image shows. They train
# auxiliary heads only and are never part of the submission.
ATOMS = [
    "acl_low_grade_or_sprain",
    "acl_reconstruction",
    "pcl_tear",
    "lcl_or_posterolateral_corner_tear",
    "mcl_low_grade_sprain",
    "medial_meniscus_degeneration_without_tear",
    "lateral_meniscus_degeneration_without_tear",
    "meniscal_displaced_fragment_or_bucket_handle",
    "cartilage_loss_low_grade_any_compartment",
    "focal_chondral_defect",
    "small_effusion",
    "loose_body",
    "extensor_tendon_abnormality",
    "patellar_dislocation_signs",
    "postoperative_hardware_or_prior_surgery",
    "subchondral_insufficiency_or_osteonecrosis",
    "degenerative_subchondral_marrow_edema",
    "parameniscal_or_ganglion_cyst",
]


def write_report_chunks(reports: pd.DataFrame, out_dir, size: int) -> list[Path]:
    """Write `StudyInstanceUID` + `Report` as JSONL chunks of `size` studies, in frame order."""
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    uids = reports["StudyInstanceUID"].astype(str)
    if uids.duplicated().any():
        raise ValueError("duplicate StudyInstanceUID in reports")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = reports["Report"].fillna("").astype(str)
    paths = []
    for start in range(0, len(reports), size):
        path = out_dir / f"chunk_{start // size:04d}.jsonl"
        lines = [json.dumps({"uid": uid, "report": text}, ensure_ascii=False)
                 for uid, text in zip(uids.iloc[start:start + size], texts.iloc[start:start + size])]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def _probability(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    value = float(value)
    return value if 0.0 <= value <= 1.0 else math.nan


def parse_label_line(line: str) -> dict | None:
    """One output row -> {"uid", 12 labels, "atom:<name>" per atom}, or None if unusable."""
    try:
        row = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(row, dict) or not isinstance(row.get("uid"), str) or not row["uid"]:
        return None
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    atoms = row.get("atoms") if isinstance(row.get("atoms"), dict) else {}
    parsed = {"uid": row["uid"]}
    parsed.update({label: _probability(labels.get(label)) for label in LABEL_COLUMNS})
    parsed.update({f"atom:{atom}": _probability(atoms.get(atom)) for atom in ATOMS})
    return parsed


def load_label_outputs(paths, expected_uids) -> pd.DataFrame:
    """Collect output files into one frame indexed by `expected_uids` (NaN where unlabelled).

    The same uid twice with different values means two runs were mixed into one pass, which
    is an error rather than something to average silently."""
    expected = [str(u) for u in expected_uids]
    allowed = set(expected)
    rows: dict[str, dict] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            parsed = parse_label_line(line)
            if parsed is None:
                continue
            uid = parsed.pop("uid")
            if uid not in allowed:
                raise ValueError(f"output uid {uid!r} is not in the expected set")
            if uid in rows and not _same(rows[uid], parsed):
                raise ValueError(f"conflicting outputs for uid {uid!r}")
            rows[uid] = parsed
    columns = list(LABEL_COLUMNS) + [f"atom:{a}" for a in ATOMS]
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    return frame.reindex(expected).astype(float)


def _same(a: dict, b: dict) -> bool:
    return all((math.isnan(a[k]) and math.isnan(b[k])) or a[k] == b[k] for k in a)
