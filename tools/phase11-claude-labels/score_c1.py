"""Gate C1: Claude's relabel of the 58 gold reports against the three-source blend.

Primary statistic (registered in NOTES before the pilot ran): cell agreement with gold, each
source thresholded at 0.5, over 58 x 12 cells; paired CI by resampling STUDIES (cells within a
study are correlated). Pass if the mean of the two Claude passes beats the blend with the CI
above zero, or the blend with Claude as a fourth source beats the blend with the CI above zero.
Per-label and macro AUC are printed beside it and do not decide.

Run from the workspace root: python3 ai/scratch/phase11-claude-labels/score_c1.py
"""
import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "source/src")
from knee.claude_labels import load_label_outputs
from knee.infer import LABEL_COLUMNS
from knee.metrics import macro_auc, per_label_auc
from knee.reports import blend_sources

D = "ai/scratch/phase11-claude-labels"
L6 = "ai/scratch/phase6-labels"
N_BOOT = 10000

train = pd.read_csv("source/data/sample/train.csv")
train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
gold = train[train[LABEL_COLUMNS].notna().any(axis=1)].set_index("StudyInstanceUID")[LABEL_COLUMNS]
assert len(gold) == 58
uids = list(gold.index)
y = gold.to_numpy(float)


def conf_source(path, use_conf):
    df = pd.read_csv(path)
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    out = pd.DataFrame(index=df["StudyInstanceUID"])
    for lab in LABEL_COLUMNS:
        v = pd.to_numeric(df[lab], errors="coerce").to_numpy()
        conf = f"{lab}__conf"
        if use_conf and conf in df.columns:
            c = pd.to_numeric(df[conf], errors="coerce").clip(0, 1).fillna(0.5).to_numpy()
            v = v * c + (1 - v) * (1 - c)
        out[lab] = v
    return out


ours = pd.read_csv("source/results/pseudo_labels_qwen3_4b.csv")
ours["StudyInstanceUID"] = ours["StudyInstanceUID"].astype(str)
ours = ours.set_index("StudyInstanceUID")[[f"score_{l}" for l in LABEL_COLUMNS]]
ours.columns = LABEL_COLUMNS
steven = conf_source(f"{L6}/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv", False)
lixin = conf_source(f"{L6}/lixin73_rsna-knee-llm-report-labels-sol56/report_labels_gpt56sol.csv", True)

passes = {}
for name in ("pass1", "pass2"):
    files = sorted(glob.glob(f"{D}/out_gold/{name}/*.jsonl"))
    frame = load_label_outputs(files, uids)[LABEL_COLUMNS]
    print(f"{name}: {frame.notna().all(axis=1).sum()}/58 studies fully labelled")
    passes[name] = frame
claude = (passes["pass1"] + passes["pass2"]) / 2

blend3 = blend_sources([steven, ours, lixin], LABEL_COLUMNS).reindex(uids)
blend4 = blend_sources([steven, ours, lixin, claude], LABEL_COLUMNS).reindex(uids)
sources = {"steven_v4": steven.reindex(uids), "ours": ours.reindex(uids), "lixin73": lixin.reindex(uids),
           "blend3": blend3, "claude_p1": passes["pass1"], "claude_p2": passes["pass2"],
           "claude_mean": claude, "blend3+claude": blend4}


def agree(frame):
    """Per-study agreement (fraction of the 12 cells right); an unlabelled cell counts as wrong."""
    p = frame.to_numpy(float)
    hit = (p >= 0.5) == (y >= 0.5)
    hit[np.isnan(p)] = False
    return hit.mean(axis=1)


print(f"\n{'source':16s} agree   macroAUC  " + " ".join(f"{l[:5]:>5s}" for l in LABEL_COLUMNS))
for name, frame in sources.items():
    s = frame.fillna(0.5).to_numpy(float)
    per = per_label_auc(y, s)
    print(f"{name:16s} {agree(frame).mean():.4f}  {macro_auc(y, s):.4f}    " + " ".join(f"{a:5.2f}" for a in per))

rng = np.random.default_rng(0)
base = agree(blend3)
print("\nGate C1 (paired study bootstrap, cell agreement):")
for name in ("claude_mean", "blend3+claude"):
    diff = agree(sources[name]) - base
    boots = np.array([diff[rng.integers(0, 58, 58)].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  {name:14s} - blend3: {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  {'PASS' if lo > 0 else 'no'}")
