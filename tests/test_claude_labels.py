import json

import numpy as np
import pandas as pd
import pytest

from knee.claude_labels import ATOMS, load_label_outputs, parse_label_line, write_report_chunks
from knee.infer import LABEL_COLUMNS


def _reports(n=5):
    frame = pd.DataFrame({
        "StudyInstanceUID": [f"1.2.{i}" for i in range(n)],
        "Report": [f"report text {i}" for i in range(n)],
    })
    for label in LABEL_COLUMNS:
        frame[label] = 1.0  # a gold label that must never reach a chunk
    return frame


def _line(uid="1.2.0", **overrides):
    row = {"uid": uid, "labels": {label: 0.1 for label in LABEL_COLUMNS},
           "atoms": {atom: 0.0 for atom in ATOMS}, "evidence": {}}
    row.update(overrides)
    return json.dumps(row)


def test_chunks_carry_only_uid_and_report_text(tmp_path):
    paths = write_report_chunks(_reports(5), tmp_path, size=2)

    assert [p.name for p in paths] == ["chunk_0000.jsonl", "chunk_0001.jsonl", "chunk_0002.jsonl"]
    rows = [json.loads(line) for p in paths for line in p.read_text().splitlines()]
    assert [r["uid"] for r in rows] == [f"1.2.{i}" for i in range(5)]
    assert all(set(r) == {"uid", "report"} for r in rows)
    for p in paths:
        text = p.read_text()
        assert all(f'"{label}"' not in text for label in LABEL_COLUMNS)


def test_chunker_rejects_duplicate_uids(tmp_path):
    frame = _reports(3)
    frame.loc[2, "StudyInstanceUID"] = "1.2.0"

    with pytest.raises(ValueError):
        write_report_chunks(frame, tmp_path, size=2)


def test_chunker_rejects_non_positive_size(tmp_path):
    with pytest.raises(ValueError):
        write_report_chunks(_reports(3), tmp_path, size=0)


def test_parse_valid_line():
    parsed = parse_label_line(_line(labels={**{l: 0.0 for l in LABEL_COLUMNS}, "ACL": 0.9}))

    assert parsed["uid"] == "1.2.0"
    assert parsed["ACL"] == 0.9 and parsed["MCL"] == 0.0
    assert all(parsed[f"atom:{a}"] == 0.0 for a in ATOMS)


def test_malformed_json_is_rejected_not_zero_filled():
    assert parse_label_line("{not json") is None
    assert parse_label_line(json.dumps({"labels": {}})) is None  # no uid
    assert parse_label_line(json.dumps(["uid", 1])) is None


def test_bad_values_become_nan_never_zero():
    labels = {l: 0.25 for l in LABEL_COLUMNS}
    labels["ACL"] = 1.7          # out of range
    labels["MCL"] = "yes"        # not a number
    labels["Fracture"] = True    # bools are not probabilities
    del labels["Synovitis"]      # missing
    parsed = parse_label_line(_line(labels=labels, atoms={}))

    for label in ("ACL", "MCL", "Fracture", "Synovitis"):
        assert np.isnan(parsed[label])
    assert parsed["Effusion"] == 0.25
    assert all(np.isnan(parsed[f"atom:{a}"]) for a in ATOMS)


def test_unknown_keys_are_ignored():
    parsed = parse_label_line(_line(labels={**{l: 0.5 for l in LABEL_COLUMNS}, "PCL": 1.0}))

    assert "PCL" not in parsed


def test_load_outputs_masks_missing_and_rejects_conflicting_duplicates(tmp_path):
    (tmp_path / "a.jsonl").write_text("\n".join([_line("1.2.0"), "garbage", _line("1.2.1")]) + "\n")
    frame = load_label_outputs([tmp_path / "a.jsonl"], expected_uids=["1.2.0", "1.2.1", "1.2.2"])

    assert list(frame.index) == ["1.2.0", "1.2.1", "1.2.2"]
    assert frame.loc["1.2.2"].isna().all()
    assert frame.loc["1.2.0", "ACL"] == 0.1

    (tmp_path / "b.jsonl").write_text(_line("1.2.0", labels={l: 0.9 for l in LABEL_COLUMNS}) + "\n")
    with pytest.raises(ValueError):
        load_label_outputs([tmp_path / "a.jsonl", tmp_path / "b.jsonl"], expected_uids=["1.2.0"])


def test_load_outputs_rejects_uids_outside_the_expected_set(tmp_path):
    (tmp_path / "a.jsonl").write_text(_line("9.9.9") + "\n")

    with pytest.raises(ValueError):
        load_label_outputs([tmp_path / "a.jsonl"], expected_uids=["1.2.0"])
