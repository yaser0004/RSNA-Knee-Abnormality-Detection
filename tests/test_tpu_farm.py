import json

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
