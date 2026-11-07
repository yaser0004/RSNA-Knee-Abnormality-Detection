"""Pooled-OOF and gold scoring of nine-group member sets, paired CIs against a base set."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'source/src')
from knee.metrics import macro_auc, paired_macro_auc_delta
C = 'source/checkpoints'
def load(d, pred, gold, uids='oof_uids.csv'):
    return (np.load(f'{C}/{d}/{pred}'), np.load(f'{C}/{d}/{gold}'), pd.read_csv(f'{C}/{d}/{uids}')['StudyInstanceUID'].astype(str).tolist())
M = {
 'c2': load('knee-phase7-tta','oof_pred_c2_6slot_attn_g9.npy','gold_pred_c2_6slot_attn_g9.npy'),
 'c3': load('knee-phase7-tta','oof_pred_c3_dinov2_2slot_g9.npy','gold_pred_c3_dinov2_2slot_g9.npy'),
 'f1': load('knee-phase7-tta-f1','oof_pred_f1_convnext_tiny_6slot_attn_g9.npy','gold_pred_f1_convnext_tiny_6slot_attn_g9.npy'),
 's2': load('knee-phase8-tta-s2','oof_pred_s2_c2_336_b4x2_g9.npy','gold_pred_s2_c2_336_b4x2_g9.npy'),
 't2': load('knee-phase8-tta-t2','oof_pred_t2_c2_blend3_g9.npy','gold_pred_t2_c2_blend3_g9.npy'),
 'a2': load('knee-phase8-confirm-a2','oof_pred_a2_c2_grpattn_rndwin_g9.npy','gold_pred_a2_c2_grpattn_rndwin_g9.npy','oof_uids_a2_c2_grpattn_rndwin.csv'),
 'd3': load('knee-phase8-confirm-d3','oof_pred_d3_a2blend_grpattn_rndwin_g9.npy','gold_pred_d3_a2blend_grpattn_rndwin_g9.npy','oof_uids_d3_a2blend_grpattn_rndwin.csv'),
 'd3_336': load('knee-phase9-confirm-d3-336','oof_pred_d3_336_a2blend_grpattn_rndwin_b4x2_g9.npy','gold_pred_d3_336_a2blend_grpattn_rndwin_b4x2_g9.npy','oof_uids_d3_336_a2blend_grpattn_rndwin_b4x2.csv'),
}
u0 = M['c2'][2]
assert all(v[2] == u0 for v in M.values()), 'row order differs'
ytrue = np.load(f'{C}/knee-phase9-confirm-d3-336/oof_true.npy'); gtrue = np.load(f'{C}/knee-phase9-confirm-d3-336/gold_true.npy')
def mean(names, i): return np.mean([M[n][i] for n in names], axis=0)
def show(tag, base, new):
    for tier, y, i in (('OOF', ytrue, 0), ('gold', gtrue, 1)):
        d, lo, hi = paired_macro_auc_delta(y, mean(base, i), mean(new, i))
        print(f'{tag:34s} {tier:4s} {macro_auc(y, mean(base,i)):.4f} -> {macro_auc(y, mean(new,i)):.4f}  {d:+.4f} [{lo:+.4f}, {hi:+.4f}]')
seven = ['c2','c3','f1','s2','t2','a2','d3']
for n in M: print(f'{n:8s} OOF {macro_auc(ytrue, M[n][0]):.4f} gold {macro_auc(gtrue, M[n][1]):.4f}')
show('d3_336 added to the seven', seven, seven + ['d3_336'])
show('d3_336 replacing d3', seven, [n for n in seven if n != 'd3'] + ['d3_336'])
show('eight vs seven (same as above)', seven, seven + ['d3_336'])
