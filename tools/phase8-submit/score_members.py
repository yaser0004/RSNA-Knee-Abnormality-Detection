"""Score member sets offline from the saved nine-group OOF/gold arrays.

    python score_members.py                # every member whose arrays are on disk
    python score_members.py +t2            # the rest as the base, t2 added and measured against it

Arrays come from the TTA kernels (and, for a2, from its own confirm, which reads both group
counts in one session). Rows are matched by StudyInstanceUID and the truth arrays are asserted
identical before anything is averaged.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from knee.metrics import macro_auc, paired_macro_auc_delta, per_label_auc  # noqa: E402
from knee.infer import LABEL_COLUMNS  # noqa: E402

CK = REPO / "checkpoints"
MEMBERS = {
    "c2": (CK / "knee-phase7-tta", "c2_6slot_attn"),
    "c3": (CK / "knee-phase7-tta", "c3_dinov2_2slot"),
    "f1": (CK / "knee-phase7-tta-f1", "f1_convnext_tiny_6slot_attn"),
    "s2": (CK / "knee-phase8-tta-s2", "s2_c2_336_b4x2"),
    "t2": (CK / "knee-phase8-tta-t2", "t2_c2_blend3"),
    # the a2 confirm reads five and nine groups in its own session, so its arrays live with it
    "a2": (CK / "knee-phase8-confirm-a2", "a2_c2_grpattn_rndwin"),
}


def available():
    return [n for n, (d, name) in MEMBERS.items()
            if (d / "oof_uids.csv").exists() and (d / f"oof_pred_{name}_g9.npy").exists()]


def load(member, groups=9):
    d, name = MEMBERS[member]
    uids = pd.read_csv(d / "oof_uids.csv")["StudyInstanceUID"].astype(str).tolist()
    return (uids, np.load(d / "oof_true.npy"), np.load(d / f"oof_pred_{name}_g{groups}.npy"),
            np.load(d / "gold_true.npy"), np.load(d / f"gold_pred_{name}_g{groups}.npy"))


def member_set(names, groups=9):
    ref_uids, oof_true, _, gold_true, _ = load(names[0], groups)
    oof, gold = [], []
    for n in names:
        uids, t, p, gt, gp = load(n, groups)
        assert uids == ref_uids, f"{n}: OOF rows differ"
        assert np.array_equal(np.nan_to_num(t, nan=-1), np.nan_to_num(oof_true, nan=-1)), f"{n}: truth differs"
        assert np.array_equal(gt, gold_true), f"{n}: gold truth differs"
        oof.append(p)
        gold.append(gp)
    return oof_true, np.mean(oof, axis=0), gold_true, np.mean(gold, axis=0)


def report(label, a, b, truth):
    d, lo, hi = paired_macro_auc_delta(truth, a, b)
    verdict = "CI excludes zero" if lo > 0 or hi < 0 else "CI spans zero"
    print(f"  {label:34s} {macro_auc(truth, a):.4f} -> {macro_auc(truth, b):.4f}  "
          f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]  {verdict}")


def main(argv):
    added = [a.lstrip("+") for a in argv if a.lstrip("+") in MEMBERS]
    have = available()
    missing = [a for a in added if a not in have]
    assert not missing, f"no arrays on disk for {missing}"
    base = [n for n in have if n not in added]
    assert base, "nothing left in the base set"

    oof_true, b_oof, gold_true, b_gold = member_set(base)
    print(f"{'+'.join(base)} at nine groups: pooled OOF {macro_auc(oof_true, b_oof):.4f} "
          f"(n={len(oof_true)}), gold {macro_auc(gold_true, b_gold):.4f} (n={len(gold_true)})")
    if not added:
        return

    for name in added:
        _, a5, _, g5 = member_set([name], groups=5)
        _, a9, _, g9 = member_set([name], groups=9)
        print(f"\n{name} alone, five -> nine groups:")
        report("OOF", a5, a9, oof_true)
        report("gold", g5, g9, gold_true)

    full = base + added
    _, n_oof, _, n_gold = member_set(full)
    print(f"\n{'+'.join(full)} against {'+'.join(base)}:")
    report("OOF", b_oof, n_oof, oof_true)
    report("gold", b_gold, n_gold, gold_true)

    print(f"\n{'label':20s} {'base OOF':>9s} {'new OOF':>9s} {'base gold':>9s} {'new gold':>9s}")
    for l, a, b, c, d in zip(LABEL_COLUMNS, per_label_auc(oof_true, b_oof), per_label_auc(oof_true, n_oof),
                             per_label_auc(gold_true, b_gold), per_label_auc(gold_true, n_gold)):
        print(f"{l:20s} {a:9.4f} {b:9.4f} {c:9.4f} {d:9.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
