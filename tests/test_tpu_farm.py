import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from knee.dataset import PreppedSlotDataset
from knee.dicom import SLICE_GROUP, SLOTS
from knee.infer import LABEL_COLUMNS
from knee.model import KneeModel
from knee.prep import save_study_npz
from knee.tpu_farm import load_inputs, save_inputs, train_job

SLOT_NAMES = [name for name, _, _ in SLOTS]
SETTINGS = dict(batch=4, epochs=1, n_groups=5, out_size=32, workers=0, pretrained=False)


def _study(root, uid, rng, size=32, n_groups=9):
    filled = list(rng.choice(SLOT_NAMES, size=int(rng.integers(1, 4)), replace=False))
    slices = {n: [rng.integers(0, 255, (size, size), dtype=np.uint8) for _ in range(n_groups * SLICE_GROUP)]
              for n in filled}
    meta = {"StudyInstanceUID": uid, "side": "L", "route": "Laterality", "is_gold": False,
            "slots": {n: (f"s-{n}" if n in filled else None) for n in SLOT_NAMES},
            "series": {f"s-{n}": {"slot": n, "n_slices_stored": len(slices[n])} for n in filled}}
    save_study_npz(root / f"{uid}.npz", slices, meta)


@pytest.fixture
def farm(tmp_path):
    rng = np.random.default_rng(0)
    uids = [f"u{i:02d}" for i in range(30)]
    root = tmp_path / "prepped"
    root.mkdir()
    for uid in uids:
        _study(root, uid, rng)
    labels = pd.DataFrame(rng.random((30, 12)), columns=LABEL_COLUMNS)
    labels.insert(0, "StudyInstanceUID", uids)
    labels.iloc[3, 2] = np.nan
    binary = labels.copy()
    binary[LABEL_COLUMNS] = (labels[LABEL_COLUMNS].to_numpy() >= 0.5).astype(float)
    holdout = {"u00", "u01", "u02"}
    inputs_dir = tmp_path / "inputs"
    save_inputs(
        inputs_dir, jobs=[dict(name="t_s0", fold=0, seed=0, target="blend")],
        targets={"blend": dict(df=labels, columns=LABEL_COLUMNS, weights=None)},
        eval_labels=binary, gold_labels=binary[binary.StudyInstanceUID.isin(holdout)],
        folds={u: i % 5 for i, u in enumerate(uids)}, holdout=holdout, present=set(uids),
        gold_uids=sorted(holdout), slot_names=SLOT_NAMES, npz_root=root, settings=SETTINGS)
    return tmp_path, inputs_dir, root, uids, holdout


def test_a_job_runs_end_to_end_and_its_checkpoint_works_in_the_ordinary_model(farm):
    tmp_path, inputs_dir, root, uids, holdout = farm
    out = tmp_path / "out"
    out.mkdir()
    inputs = load_inputs(inputs_dir)
    summary = train_job(inputs["jobs"][0], inputs, {**SETTINGS}, out)

    val_uids = pd.read_csv(out / "val_uids_t_s0.csv")["StudyInstanceUID"].astype(str).tolist()
    assert summary["n_val"] == len(val_uids) and not set(val_uids) & holdout
    for n in (5, 9):
        pred = np.load(out / f"val_pred_t_s0_g{n}.npy")
        assert pred.shape == (len(val_uids), 12) and np.isfinite(pred).all() and ((pred > 0) & (pred < 1)).all()
        assert np.load(out / f"gold_pred_t_s0_g{n}.npy").shape == (len(holdout), 12)

    # the saved weights are an ordinary checkpoint: a normal packed KneeModel reproduces the farm's eval
    model = KneeModel(backbone_name="efficientnet_b0", num_labels=12, pretrained=False, head="group_attention")
    model.load_state_dict(torch.load(out / "t_s0_fold0.pt", weights_only=True))
    model.eval()
    ds = PreppedSlotDataset(val_uids, root, labels_df=inputs["eval_labels"], n_groups=9, out_size=32,
                            slots=SLOT_NAMES, augment=False)
    with torch.no_grad():
        images, mask = zip(*[(ds[i][0], ds[i][1]) for i in range(len(val_uids))])
        ref = torch.sigmoid(model(torch.stack(images), mask=torch.stack(mask))).numpy()
    assert np.allclose(ref, np.load(out / "val_pred_t_s0_g9.npy"), atol=1e-4)


def test_inputs_round_trip_and_bad_inputs_are_rejected(farm):
    tmp_path, inputs_dir, *_ = farm
    inputs = load_inputs(inputs_dir)
    assert inputs["jobs"][0]["name"] == "t_s0" and inputs["targets"]["blend"]["weights"] is None
    assert isinstance(inputs["holdout"], frozenset) and len(inputs["folds"]) == 30

    spec = json.loads((inputs_dir / "inputs.json").read_text())
    spec["jobs"].append(dict(name="t_s0", fold=1, seed=0, target="blend"))
    (inputs_dir / "inputs.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="unique"):
        load_inputs(inputs_dir)
    spec["jobs"] = [dict(name="x", fold=1, seed=0, target="nope")]
    (inputs_dir / "inputs.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="unknown target"):
        load_inputs(inputs_dir)


def test_a_fold_with_fewer_than_one_batch_fails_loudly(farm):
    tmp_path, inputs_dir, *_ = farm
    out = tmp_path / "out"
    out.mkdir()
    inputs = load_inputs(inputs_dir)
    with pytest.raises(ValueError, match="full batch"):
        train_job(inputs["jobs"][0], inputs, {**SETTINGS, "batch": 64}, out)


# --- the per-chip launcher: torch_xla.launch does not work on Kaggle's TPU VM ----------------
# (probe 2: "Expected 8 worker addresses, got 1"), so workers are plain processes, one chip each.

STUB = '''
import os, sys, time
from pathlib import Path
inputs, out, rank = sys.argv[-3:]
chips = os.environ.get("TPU_VISIBLE_CHIPS")
has_ports = "TPU_PROCESS_PORT" in os.environ
one_chip = os.environ.get("TPU_PROCESS_BOUNDS") == "1,1,1"
mode = os.environ["STUB_WORKS_WHEN"]
works = {"one_chip": one_chip and not has_ports, "ports": one_chip and has_ports, "default": chips is None,
         "never": False}[mode]
if not works:
    sys.exit(3)
Path(out, f"ready_{rank}").write_text(f"chips={chips}")
Path(out, f"ran_{rank}").write_text("x")
'''


def _stub_cmd(tmp_path):
    script = tmp_path / "stub.py"
    script.write_text(STUB)
    return lambda inputs, out, rank: [sys.executable, str(script), inputs, out, str(rank)]


def test_chip_env_gives_each_rank_its_own_chip_and_port():
    from knee.tpu_farm import CHIP_VARIANTS, chip_env
    for variant in CHIP_VARIANTS:
        envs = [chip_env(variant, r) for r in range(8)]
        assert sorted(e["TPU_VISIBLE_CHIPS"] for e in envs) == [str(i) for i in range(8)]
    full = [chip_env("one_chip_ports", r) for r in range(8)]
    assert len({e["TPU_PROCESS_PORT"] for e in full}) == 8
    assert all(e["TPU_PROCESS_BOUNDS"] == "1,1,1" and e["TPU_PROCESS_ADDRESSES"].endswith(e["TPU_PROCESS_PORT"])
               for e in full)
    assert "TPU_PROCESS_PORT" not in chip_env("one_chip", 3) and "TPU_PROCESS_BOUNDS" not in chip_env("visible_only", 3)
    with pytest.raises(ValueError):
        chip_env("one_chip", 8)
    with pytest.raises(ValueError):
        chip_env("nonsense", 0)


@pytest.mark.parametrize("works_when, expected_variant", [("ports", "one_chip_ports"), ("one_chip", "one_chip")])
def test_launcher_moves_to_the_next_variant_until_workers_come_up(tmp_path, monkeypatch, works_when, expected_variant):
    from knee.tpu_farm import launch_per_chip
    monkeypatch.setenv("STUB_WORKS_WHEN", works_when)
    out = tmp_path / "out"
    out.mkdir()
    report = launch_per_chip("in", out, n_jobs=3, worker_cmd=_stub_cmd(tmp_path), ready_timeout=20)

    assert report[-1]["variant"] == expected_variant and report[-1]["up"] == [0, 1, 2]
    assert [p.name for p in sorted(out.glob("ran_*"))] == ["ran_0", "ran_1", "ran_2"]
    assert [p.read_text() for p in sorted(out.glob("ready_*"))] == ["chips=0", "chips=1", "chips=2"]


def test_launcher_falls_back_to_a_few_default_environment_jobs_when_no_variant_works(tmp_path, monkeypatch):
    from knee.tpu_farm import launch_per_chip
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("STUB_WORKS_WHEN", "never")
    report = launch_per_chip("in", out, n_jobs=5, worker_cmd=_stub_cmd(tmp_path), ready_timeout=20, fallback_jobs=2)
    assert report[-1]["variant"] == "single_process_default" and report[-1]["up"] == []
    assert not list(out.glob("ran_*"))

    monkeypatch.setenv("STUB_WORKS_WHEN", "default")
    out2 = tmp_path / "out2"
    out2.mkdir()
    report = launch_per_chip("in", out2, n_jobs=5, worker_cmd=_stub_cmd(tmp_path), ready_timeout=20, fallback_jobs=2)
    assert report[-1]["variant"] == "single_process_default" and report[-1]["up"] == [0, 1]
    assert sorted(p.name for p in out2.glob("ran_*")) == ["ran_0", "ran_1"]  # capped: not all five


def test_a_hung_worker_is_killed_after_the_run_timeout(tmp_path, monkeypatch):
    from knee.tpu_farm import launch_per_chip
    hang = tmp_path / "hang.py"
    hang.write_text("import sys, time\nfrom pathlib import Path\nout, rank = sys.argv[-2:]\n"
                    "Path(out, f'ready_{rank}').write_text('1')\ntime.sleep(120 if rank == '1' else 0)\n")
    out = tmp_path / "out"
    out.mkdir()
    cmd = lambda inputs, o, rank: [sys.executable, str(hang), o, str(rank)]
    start = __import__("time").time()
    report = launch_per_chip("in", out, n_jobs=2, worker_cmd=cmd, ready_timeout=20, run_timeout=3)
    assert report[-1]["up"] == [0, 1] and __import__("time").time() - start < 60
    assert (out / "timeout_1.txt").exists() and not (out / "timeout_0.txt").exists()
