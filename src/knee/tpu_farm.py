"""One TPU session, up to eight independent training jobs: one job per chip, no gradient sharing.

Each job is the GPU recipe (`notebooks/phase8-confirm-d3`) for one (fold, seed, target): the same
model, optimiser, schedule and batch, so a TPU-trained fold is a drop-in for a GPU-trained one.
What changes is only what XLA requires: fixed shapes (masked BatchNorm in place of packing absent
slots, a static loss, padded evaluation batches), device-tensor optimiser hyperparameters, and
bfloat16 autocast. Chips do not talk to each other, so a failed job costs only itself.

Run on the TPU VM as `python -m knee.tpu_farm <inputs_dir> <out_dir>` (workers: `--worker <inputs_dir> <out_dir> <rank>`). Without torch_xla the same
code runs the jobs one after another on CPU, which is what the tests do.
"""
import contextlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from knee.dataset import PreppedSlotDataset
from knee.infer import LABEL_COLUMNS
from knee.metrics import macro_auc
from knee.model import KneeModel, keep_labels
from knee.train import train_val_split
from knee.xla_train import TensorAdam, masked_bce_loss_static, one_cycle_schedule, pad_to_multiple

try:
    import torch_xla
    import torch_xla.distributed.parallel_loader as parallel_loader
    XLA = True
except ImportError:
    XLA = False

DEFAULTS = dict(batch=8, epochs=8, n_groups=5, out_size=224, workers=6, max_lr=3e-4, pretrained=True,
                backbone="efficientnet_b0")


def _device():
    return torch_xla.device() if XLA else torch.device("cpu")


def _sync():
    if XLA:
        torch_xla.sync()


def _autocast():
    return torch.autocast("xla", dtype=torch.bfloat16) if XLA else contextlib.nullcontext()


def _on_device(loader, device):
    return parallel_loader.MpDeviceLoader(loader, device) if XLA else loader


class _DropUid(Dataset):
    """The dataset's trailing study uid is a string, which cannot ride to the device."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][:-1]


def _slot_dataset(uids, inputs, labels_df, n_groups, settings, train, columns=None, weights_df=None):
    kwargs = dict(labels_df=labels_df, n_groups=n_groups, out_size=settings["out_size"],
                  slots=inputs["slot_names"], augment=train, random_groups=train)
    if columns is not None:
        kwargs.update(label_columns=columns, weights_df=weights_df)
    return _DropUid(PreppedSlotDataset(uids, inputs["npz_root"], **kwargs))


def predict(model, device, uids, inputs, labels_df, n_groups, settings, n_scored=len(LABEL_COLUMNS)):
    """(truth, probability) for `uids` in order, first `n_scored` outputs. Batches keep one shape."""
    padded, n_real = pad_to_multiple(uids, settings["batch"])
    dataset = _slot_dataset(padded, inputs, labels_df, n_groups, settings, train=False)
    loader = _on_device(DataLoader(dataset, batch_size=settings["batch"], shuffle=False,
                                   num_workers=settings["workers"]), device)
    model.eval()
    truths, probs = [], []
    with torch.no_grad():
        for images, mask, labels in loader:
            with _autocast():
                logits = model(images, mask=mask)
            probs.append(torch.sigmoid(logits.float()[:, :n_scored]).cpu())
            truths.append(labels.cpu())
    return torch.cat(truths)[:n_real].numpy(), torch.cat(probs)[:n_real].numpy()


def train_job(job: dict, inputs: dict, settings: dict, out_dir) -> dict:
    """Train one fold of one target and write its checkpoint, predictions and a summary."""
    out_dir = Path(out_dir)
    settings = {**DEFAULTS, **settings}
    name, fold, seed = job["name"], job["fold"], job["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _device()
    target = inputs["targets"][job["target"]]
    present = inputs["present"]

    train_uids, val_uids = train_val_split(inputs["folds"], val_fold=fold, exclude_uids=inputs["holdout"])
    train_uids = [u for u in train_uids if u in present]
    val_uids = [u for u in val_uids if u in present]
    gold_uids = [u for u in inputs["gold_uids"] if u in present]
    if not train_uids or not val_uids:
        raise ValueError(f"{name}: empty split (train {len(train_uids)}, val {len(val_uids)})")

    columns = target["columns"]
    train_set = _slot_dataset(train_uids, inputs, target["df"], settings["n_groups"], settings, train=True,
                              columns=columns, weights_df=target.get("weights"))
    loader = DataLoader(train_set, batch_size=settings["batch"], shuffle=True,
                        num_workers=settings["workers"], drop_last=True)
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise ValueError(f"{name}: fewer than one full batch of {settings['batch']} training studies")
    loader = _on_device(loader, device)

    model = KneeModel(backbone_name=settings["backbone"], num_labels=len(columns),
                      pretrained=settings["pretrained"], head="group_attention")
    model.enable_static_shapes()
    model.to(device)
    optimizer = TensorAdam(model.parameters())
    lrs, beta1s = one_cycle_schedule(settings["epochs"] * steps_per_epoch, settings["max_lr"])

    epoch_seconds, step, started = [], 0, time.time()
    for epoch in range(settings["epochs"]):
        model.train()
        tick, total = time.time(), torch.zeros((), device=device)
        for batch in loader:
            images, mask, labels, *rest = batch
            with _autocast():
                logits = model(images, mask=mask)
            loss = masked_bce_loss_static(logits.float(), labels, rest[0] if rest else None)
            loss.backward()
            optimizer.step(lr=torch.tensor(lrs[step], device=device), beta1=torch.tensor(beta1s[step], device=device))
            optimizer.zero_grad()
            _sync()
            total += loss.detach()
            step += 1
        epoch_seconds.append(round(time.time() - tick, 1))
        print(f"[{name}] epoch {epoch + 1}/{settings['epochs']} loss {float(total) / steps_per_epoch:.4f} "
              f"{epoch_seconds[-1]}s", flush=True)
    minutes = (time.time() - started) / 60

    n_scored = len(LABEL_COLUMNS)
    reads = {}
    for n_groups in (5, 9):
        truth, prob = predict(model, device, val_uids, inputs, inputs["eval_labels"], n_groups, settings, n_scored)
        g_truth, g_prob = predict(model, device, gold_uids, inputs, inputs["gold_labels"], n_groups, settings, n_scored)
        reads[n_groups] = (truth, prob, g_truth, g_prob)
        np.save(out_dir / f"val_pred_{name}_g{n_groups}.npy", prob)
        np.save(out_dir / f"gold_pred_{name}_g{n_groups}.npy", g_prob)
    np.save(out_dir / f"val_true_{name}.npy", reads[9][0])
    np.save(out_dir / "gold_true.npy", reads[9][2])
    pd.DataFrame({"StudyInstanceUID": val_uids}).to_csv(out_dir / f"val_uids_{name}.csv", index=False)

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(keep_labels(state, n_scored), out_dir / f"{name}_fold{fold}.pt")
    summary = dict(name=name, fold=fold, seed=seed, target=job["target"], train_minutes=round(minutes, 1),
                   epoch_seconds=epoch_seconds, n_train=len(train_uids), n_val=len(val_uids),
                   val_macro_g5=float(macro_auc(reads[5][0], reads[5][1])),
                   val_macro_g9=float(macro_auc(reads[9][0], reads[9][1])),
                   gold_macro_g5=float(macro_auc(reads[5][2], reads[5][3])),
                   gold_macro_g9=float(macro_auc(reads[9][2], reads[9][3])))
    (out_dir / f"summary_{name}.json").write_text(json.dumps(summary))
    print(f"[{name}] done: {json.dumps(summary)}", flush=True)
    return summary


def save_inputs(directory, *, jobs, targets, eval_labels, gold_labels, folds, holdout, present, gold_uids,
                slot_names, npz_root, settings=None) -> None:
    """Write everything the workers need as JSON and CSV (no pickle: nothing here is executable).

    `targets` maps a name to dict(df=DataFrame with StudyInstanceUID, columns=[...], weights=DataFrame|None)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec = {}
    for name, target in targets.items():
        target["df"].to_csv(directory / f"target_{name}.csv", index=False)
        has_weights = target.get("weights") is not None
        if has_weights:
            target["weights"].to_csv(directory / f"weights_{name}.csv", index=False)
        spec[name] = dict(columns=list(target["columns"]), weights=has_weights)
    eval_labels.to_csv(directory / "eval_labels.csv", index=False)
    gold_labels.to_csv(directory / "gold_labels.csv", index=False)
    (directory / "inputs.json").write_text(json.dumps(dict(
        jobs=jobs, targets=spec, folds={str(u): int(f) for u, f in folds.items()}, holdout=sorted(holdout),
        present=sorted(present), gold_uids=list(gold_uids), slot_names=list(slot_names), npz_root=str(npz_root),
        settings=settings or {})))


def load_inputs(directory) -> dict:
    """Inverse of save_inputs, with the shape checks a corrupt or mismatched directory would trip."""
    directory = Path(directory)
    raw = json.loads((directory / "inputs.json").read_text())
    for job in raw["jobs"]:
        missing = {"name", "fold", "seed", "target"} - set(job)
        if missing or job["target"] not in raw["targets"]:
            raise ValueError(f"bad job {job}: missing {sorted(missing)} or unknown target")
    if len({j["name"] for j in raw["jobs"]}) != len(raw["jobs"]):
        raise ValueError("job names must be unique: they name the output files")
    targets = {}
    for name, spec in raw["targets"].items():
        df = pd.read_csv(directory / f"target_{name}.csv")
        df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
        if not set(spec["columns"]) <= set(df.columns):
            raise ValueError(f"target {name}: columns missing from its table")
        weights = None
        if spec["weights"]:
            weights = pd.read_csv(directory / f"weights_{name}.csv")
            weights["StudyInstanceUID"] = weights["StudyInstanceUID"].astype(str)
        targets[name] = dict(df=df, columns=spec["columns"], weights=weights)
    labels = {}
    for key in ("eval_labels", "gold_labels"):
        labels[key] = pd.read_csv(directory / f"{key}.csv")
        labels[key]["StudyInstanceUID"] = labels[key]["StudyInstanceUID"].astype(str)
    return dict(jobs=raw["jobs"], targets=targets, folds=raw["folds"], holdout=frozenset(raw["holdout"]),
                present=frozenset(raw["present"]), gold_uids=raw["gold_uids"], slot_names=raw["slot_names"],
                npz_root=raw["npz_root"], settings=raw["settings"], **labels)


def _worker(rank, inputs_dir, out_dir):
    """Run job `rank` (CPU tests call this directly; on a TPU, `_run_one_chip` does)."""
    inputs = load_inputs(inputs_dir)
    jobs = inputs["jobs"]
    if rank >= len(jobs):
        return
    try:
        train_job(jobs[rank], inputs, inputs.get("settings", {}), out_dir)
    except Exception:
        Path(out_dir, f"error_{jobs[rank]['name']}.txt").write_text(traceback.format_exc())
        print(f"[{jobs[rank]['name']}] FAILED\n{traceback.format_exc()}", flush=True)


# torch_xla.launch (multi-process) fails on Kaggle's TPU VM: "Expected 8 worker addresses, got 1"
# (probe 2, 2026-10-08), because the container's TPU topology variables describe one process. The
# workers here are therefore independent processes, each told it owns exactly one chip.
# farm1 finished eight 21-minute jobs in 24 minutes; ~2.5x that is a hang, not a slow job
RUN_TIMEOUT = 60 * 60
CHIP_VARIANTS = ("one_chip_ports", "one_chip", "visible_only")


def chip_env(variant: str, rank: int, base_port: int = 8476, metrics_port: int = 8431) -> dict:
    """Environment that makes a process see only chip `rank` as its whole 1x1x1 slice."""
    if variant not in CHIP_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if not 0 <= rank < 8:
        raise ValueError(f"a v5e-8 host has chips 0..7, got {rank}")
    env = {"TPU_VISIBLE_CHIPS": str(rank)}
    if variant in ("one_chip_ports", "one_chip"):
        env.update(TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1", TPU_PROCESS_BOUNDS="1,1,1")
    if variant == "one_chip_ports":
        port = base_port + rank
        env.update(TPU_PROCESS_PORT=str(port), TPU_PROCESS_ADDRESSES=f"localhost:{port}", CLOUD_TPU_TASK_ID="0",
                   TPU_RUNTIME_METRICS_PORTS=str(metrics_port + rank))
    return env


def _run_one_chip(inputs_dir, out_dir, rank):
    """Worker process: bring the device up, say so, then train job `rank`."""
    torch.zeros(1, device=_device()).cpu()
    Path(out_dir, f"ready_{rank}").write_text("1")
    _worker(rank, inputs_dir, out_dir)


def _default_cmd(inputs_dir, out_dir, rank):
    return [sys.executable, "-m", "knee.tpu_farm", "--worker", str(inputs_dir), str(out_dir), str(rank)]


def launch_per_chip(inputs_dir, out_dir, n_jobs, fallback_jobs=2, ready_timeout=300, worker_cmd=_default_cmd,
                    run_timeout=None):
    """Start one worker per job, trying chip-environment variants until some workers initialise.

    A worker that cannot attach to its chip dies within a minute or two; one that attaches writes
    ready_<rank>. The first variant with any ready worker is kept and run to completion. If none
    works, up to `fallback_jobs` jobs run one after another in a single default-environment process
    (the mode that worked in probe 1), so the session still returns something. `run_timeout` (seconds, after
    the workers are up) kills stragglers: a hung worker would otherwise hold the session until Kaggle's cap,
    and a cancelled session saves no outputs at all (farm2, 2026-10-11). Returns a report."""
    out = Path(out_dir)
    report = []
    for variant in CHIP_VARIANTS:
        procs = [subprocess.Popen(worker_cmd(inputs_dir, out_dir, rank), env={**os.environ, **chip_env(variant, rank)})
                 for rank in range(n_jobs)]
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            time.sleep(1)
            up = [r for r in range(n_jobs) if (out / f"ready_{r}").exists()]
            if len(up) + sum(p.poll() is not None and not (out / f"ready_{r}").exists()
                             for r, p in enumerate(procs)) >= n_jobs:
                break
        up = [r for r in range(n_jobs) if (out / f"ready_{r}").exists()]
        report.append(dict(variant=variant, up=up))
        print(f"[launcher] {variant}: {len(up)}/{n_jobs} workers initialised", flush=True)
        if up:
            end = None if run_timeout is None else time.time() + run_timeout
            for rank, p in enumerate(procs):
                try:
                    p.wait(timeout=None if end is None else max(end - time.time(), 0))
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                    Path(out_dir, f"timeout_{rank}.txt").write_text(f"killed after {run_timeout} s")
                    print(f"[launcher] worker {rank} killed after the run timeout", flush=True)
            break
        for p in procs:
            p.kill()
            p.wait()
        time.sleep(5)
    else:
        ran = []
        for rank in range(min(fallback_jobs, n_jobs)):
            done = subprocess.run(worker_cmd(inputs_dir, out_dir, rank), env=dict(os.environ))
            if (out / f"ready_{rank}").exists():
                ran.append(rank)
            print(f"[launcher] single-process fallback, job {rank}: exit {done.returncode}", flush=True)
        report.append(dict(variant="single_process_default", up=ran))
    (out / "launcher.json").write_text(json.dumps(report))
    return report


def main(inputs_dir, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n_jobs = len(load_inputs(inputs_dir)["jobs"])
    if XLA:
        print("[launcher]", json.dumps(launch_per_chip(inputs_dir, out_dir, n_jobs, run_timeout=RUN_TIMEOUT)),
              flush=True)
    else:
        for rank in range(n_jobs):
            _worker(rank, inputs_dir, out_dir)


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        _run_one_chip(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        main(sys.argv[1], sys.argv[2])
