"""Read farm3 (round 2): pool c2 over five folds and compare with round 1, by the rules registered in NOTES.

usage (workspace root): python3 ai/scratch/phase11-tpu/read_farm3.py
Needs source/checkpoints/oof_{b,c,d}_s0_g9.npy and gold_{b,c,d}_s0_g9.npy from read_farm2.py.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "source/src")
from knee.infer import LABEL_COLUMNS
from knee.metrics import macro_auc, paired_macro_auc_delta, per_label_auc

C = "source/checkpoints"
F3 = f"{C}/knee-phase11-tpu-farm3/farm_out"
D3 = f"{C}/knee-phase8-confirm-d3"
d3_uids = pd.read_csv(f"{D3}/oof_uids_d3_a2blend_grpattn_rndwin.csv")["StudyInstanceUID"].astype(str).tolist()
d3_pred = np.load(f"{D3}/oof_pred_d3_a2blend_grpattn_rndwin_g9.npy")
d3_true = np.load(f"{D3}/oof_true_d3_a2blend_grpattn_rndwin.npy")
g_true = np.load(f"{F3}/gold_true.npy")
row = {u: i for i, u in enumerate(d3_uids)}

for e in sorted(glob.glob(f"{F3}/error_*.txt")) + sorted(glob.glob(f"{F3}/timeout_*.txt")):
    print("PROBLEM", e, open(e).read()[-600:])
summ = {json.load(open(p))["name"]: json.load(open(p)) for p in sorted(glob.glob(f"{F3}/summary_*.json"))}
print(f"{'job':10s} {'min':>6s} {'val g9':>7s} {'gold g9':>8s}")
for n, s in summ.items():
    print(f"{n:10s} {s['train_minutes']:6.1f} {s['val_macro_g9']:7.4f} {s['gold_macro_g9']:8.4f}")


def load(job):
    uids = pd.read_csv(f"{F3}/val_uids_{job}.csv")["StudyInstanceUID"].astype(str).tolist()
    return uids, np.load(f"{F3}/val_pred_{job}_g9.npy"), np.load(f"{F3}/val_true_{job}.npy"), \
        np.load(f"{F3}/gold_pred_{job}_g9.npy")


def boot(a, b):
    return paired_macro_auc_delta(g_true, a, b)


# 1. c2 five folds pooled
jobs = [f"c2_s0_f{k}" for k in range(5)]
assert all(j in summ for j in jobs), f"missing c2 folds: {[j for j in jobs if j not in summ]}"
oof2 = np.full_like(d3_pred, np.nan)
golds = []
for j in jobs:
    uids, pred, true, gold = load(j)
    idx = [row[u] for u in uids]
    assert np.array_equal(np.nan_to_num(d3_true[idx], nan=-1), np.nan_to_num(true, nan=-1))
    oof2[idx] = pred
    golds.append(gold)
assert not np.isnan(oof2).any()
gold2 = np.mean(golds, axis=0)
np.save(f"{C}/oof_c2_s0_g9.npy", oof2)
np.save(f"{C}/gold_c2_s0_g9.npy", gold2)
oof_c, gold_c = np.load(f"{C}/oof_c_s0_g9.npy"), np.load(f"{C}/gold_c_s0_g9.npy")
oof_b, gold_b = np.load(f"{C}/oof_b_s0_g9.npy"), np.load(f"{C}/gold_b_s0_g9.npy")
oof_d, gold_d = np.load(f"{C}/oof_d_s0_g9.npy"), np.load(f"{C}/gold_d_s0_g9.npy")
print(f"\n1. c2 x5: pooled OOF vs steven {macro_auc(d3_true, oof2):.4f} (c {macro_auc(d3_true, oof_c):.4f}); "
      f"gold of 5-fold mean {macro_auc(g_true, gold2):.4f} (c {macro_auc(g_true, gold_c):.4f}) -> gate threshold 0.9108")
d, lo, hi = boot(gold_c, gold2)
print(f"   gold c2 - c (5-fold mean): {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
print("   per-label gold c -> c2:", " ".join(f"{l[:4]}:{a:.2f}->{b:.2f}" for l, a, b in
                                         zip(LABEL_COLUMNS, per_label_auc(g_true, gold_c), per_label_auc(g_true, gold2))))
gate = macro_auc(g_true, gold2) >= 0.9108
print(f"   REGISTERED GATE (c2 five-fold gold mean >= 0.9108): {'PASS' if gate else 'FAIL'}")

# 2. fold 0, c2 vs c (single model, the one that would be submitted)
c0 = np.load(f"{C}/knee-phase11-tpu-farm1/farm_out/gold_pred_c_s0_f0_g9.npy")
_, p2, t2, g2 = load("c2_s0_f0")
d, lo, hi = boot(c0, g2)
print(f"2. fold 0 gold c_s0_f0 {macro_auc(g_true, c0):.4f} -> c2_s0_f0 {macro_auc(g_true, g2):.4f} ({d:+.4f} [{lo:+.4f}, {hi:+.4f}])")

# 3. seed spread
if "c2_s1_f0" in summ:
    _, _, _, g21 = load("c2_s1_f0")
    d, lo, hi = boot(g2, g21)
    print(f"3. seed spread c2 f0 gold {macro_auc(g_true, g2):.4f} vs seed1 {macro_auc(g_true, g21):.4f} ({d:+.4f} [{lo:+.4f}, {hi:+.4f}])")

# 4. d2 / b2 against c2 on fold 0
for arm in ("d2_s0_f0", "b2_s0_f0"):
    if arm in summ:
        _, _, _, ga = load(arm)
        d, lo, hi = boot(g2, ga)
        print(f"4. {arm} vs c2_s0_f0 gold {macro_auc(g_true, g2):.4f} -> {macro_auc(g_true, ga):.4f} ({d:+.4f} [{lo:+.4f}, {hi:+.4f}])")

# 5. does round 2 move toward the ensemble or just copy the teacher?
t2arr = pd.read_csv("ai/scratch/phase11-student/teacher_ds/teacher2_oof.csv").set_index("StudyInstanceUID")[LABEL_COLUMNS]
t2m = t2arr.reindex(d3_uids).to_numpy()
corr = float(np.mean([np.corrcoef(oof2[:, j], t2m[:, j])[0, 1] for j in range(12)]))
corr_c = float(np.mean([np.corrcoef(oof_c[:, j], t2m[:, j])[0, 1] for j in range(12)]))
print(f"5. mean per-label correlation of OOF with teacher2: c2 {corr:.3f} vs c {corr_c:.3f}")
mean_r1 = np.mean([gold_b, gold_c, gold_d], axis=0)
print(f"6. gold: round-1 b+c+d mean {macro_auc(g_true, mean_r1):.4f}; c2 alone {macro_auc(g_true, gold2):.4f}; "
      f"round-1 mean + c2 {macro_auc(g_true, np.mean([gold_b, gold_c, gold_d, gold2], axis=0)):.4f}")
