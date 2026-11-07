"""Gate G1: five-fold heads on cached frozen features. Run from source/.

usage: python3 ../ai/scratch/phase10-g1/g1.py <cache.npz> <out_dir> [epochs]
Rules and arms: NOTES.md 'Gate G1 harness' (2026-10-06). Writes per-arm OOF/gold arrays and prints the readout."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

sys.path.insert(0, 'src')
from knee.feature_head import FeatureHead
from knee.features import load_feature_cache
from knee.infer import LABEL_COLUMNS
from knee.metrics import macro_auc, paired_macro_auc_delta, per_label_auc
from knee.reports import blend_sources
from knee.train import masked_bce_loss

CACHE, OUT = sys.argv[1], Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
P6 = Path('../ai/scratch/phase6-labels')
CK = Path('checkpoints')
dev = 'cuda' if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 2.5e9 else 'cpu'
print('device', dev, flush=True)

# ---- cache, folds, gold, targets (as the Phase 8 confirm kernels build them) -----------------------------
cache = load_feature_cache(CACHE)
cuid = [str(u) for u in cache.uids]; row = {u: i for i, u in enumerate(cuid)}
folds = pd.read_csv('results/folds_primary_v3.csv').set_index('StudyInstanceUID')['fold'].to_dict()
train_df = pd.read_csv('data/sample/train.csv')
gold_df = train_df[train_df[[c for c in train_df.columns if c not in ('StudyInstanceUID', 'Report')]].notna().any(axis=1)]
gold_uids = [u for u in gold_df['StudyInstanceUID'].astype(str) if u in row]
holdout = set(gold_df['StudyInstanceUID'].astype(str))

def conf_source(path, use_conf):
    df = pd.read_csv(path); df['StudyInstanceUID'] = df['StudyInstanceUID'].astype(str)
    out = pd.DataFrame({'StudyInstanceUID': df['StudyInstanceUID']})
    for lab in LABEL_COLUMNS:
        v = pd.to_numeric(df[lab], errors='coerce'); cc = f'{lab}__conf'
        if use_conf and cc in df.columns:
            c = pd.to_numeric(df[cc], errors='coerce').clip(0, 1).fillna(0.5); v = v * c + (1 - v) * (1 - c)
        out[lab] = v
    return out.set_index('StudyInstanceUID')
steven = conf_source(P6 / 'stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv', False)
lixin = conf_source(P6 / 'lixin73_rsna-knee-llm-report-labels-sol56/report_labels_gpt56sol.csv', True)
ours = pd.read_csv('results/pseudo_labels_qwen3_4b.csv'); ours['StudyInstanceUID'] = ours['StudyInstanceUID'].astype(str)
ours = ours.set_index('StudyInstanceUID')[[f'score_{l}' for l in LABEL_COLUMNS]]; ours.columns = LABEL_COLUMNS
blend = blend_sources([steven, ours, lixin], LABEL_COLUMNS)
truth_frame = (steven[LABEL_COLUMNS].to_numpy(float) >= 0.5).astype(float)
truth_frame[np.isnan(steven[LABEL_COLUMNS].to_numpy(float))] = np.nan
truth_frame = pd.DataFrame(truth_frame, index=steven.index, columns=LABEL_COLUMNS)

labeled = sorted(u for u in cuid if u in folds and u not in holdout)           # train_val_split's order
Y = torch.tensor(blend.reindex(cuid)[LABEL_COLUMNS].to_numpy(np.float32))       # soft target, NaN = absent
T = truth_frame.reindex(labeled)[LABEL_COLUMNS].to_numpy(float)                 # fixed truth, scored on
G = gold_df.set_index(gold_df['StudyInstanceUID'].astype(str)).loc[gold_uids, LABEL_COLUMNS].to_numpy(float)
print(f'cache {cache.features.shape} | labeled {len(labeled)} | gold {len(gold_uids)}', flush=True)

FEAT = torch.from_numpy(cache.features)        # fp16 [N, S, G, D] on CPU
MASK = torch.from_numpy(cache.mask)
D = FEAT.shape[-1]

class Net(nn.Module):
    def __init__(self, adapter):
        super().__init__()
        self.drop = nn.Dropout(0.2)
        self.adapter = nn.Sequential(nn.Linear(D, 512), nn.GELU(), nn.Dropout(0.3)) if adapter else nn.Identity()
        self.head = FeatureHead(512 if adapter else D)
    def forward(self, x, mask):
        return self.head(self.adapter(self.drop(x)), mask)

def batch(idx, mu, sd, windows=None):
    x = FEAT[idx].to(dev).float(); m = MASK[idx].to(dev)
    x = (x - mu) / sd * m[:, :, None, None]
    if windows == 'random':
        B, S, Gw, _ = x.shape
        pick = torch.rand(B, Gw, device=dev).argsort(1)[:, :5].sort(1).values
        x = torch.gather(x, 2, pick[:, None, :, None].expand(B, S, 5, x.shape[-1]))
    elif windows is not None:
        x = x[:, :, windows]
    return x, m

def fit_stats(idx):
    s = torch.zeros(D, device=dev); ss = torch.zeros(D, device=dev); n = 0
    for i in range(0, len(idx), 256):
        x = FEAT[idx[i:i + 256]].to(dev).float(); m = MASK[idx[i:i + 256]].to(dev)
        tok = x[m]                                  # present slots only: [n_slots_present, G, D]
        s += tok.sum((0, 1)); ss += (tok ** 2).sum((0, 1)); n += tok.shape[0] * tok.shape[1]
    mu = s / n; sd = (ss / n - mu ** 2).clamp(min=1e-6).sqrt()
    return mu[None, None, None, :], sd[None, None, None, :]

@torch.no_grad()
def predict(net, idx, mu, sd, windows):
    net.eval(); out = []
    for i in range(0, len(idx), 128):
        x, m = batch(idx[i:i + 128], mu, sd, windows)
        out.append(torch.sigmoid(net(x, m)).cpu().numpy())
    return np.concatenate(out)

FIVE = torch.tensor([0, 2, 4, 6, 8])
res = {}
for arm, adapter in (('A_linear', False), ('B_adapter', True)):
    oof9 = np.zeros((len(labeled), 12), np.float32); oof5 = np.zeros_like(oof9)
    gold9 = np.zeros((len(gold_uids), 12), np.float32); gold5 = np.zeros_like(gold9)
    pos = {u: i for i, u in enumerate(labeled)}
    gidx = torch.tensor([row[u] for u in gold_uids])
    t0 = time.time()
    for f in range(5):
        tr = [row[u] for u in labeled if folds[u] != f]; va = [u for u in labeled if folds[u] == f]
        tr = torch.tensor(tr); vai = torch.tensor([row[u] for u in va])
        torch.manual_seed(f); mu, sd = fit_stats(tr)
        net = Net(adapter).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        steps = -(-len(tr) // 64)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, epochs=EPOCHS, steps_per_epoch=steps)
        for ep in range(EPOCHS):
            net.train(); perm = tr[torch.randperm(len(tr))]; tot = 0
            for i in range(0, len(perm), 64):
                b = perm[i:i + 64]
                x, m = batch(b, mu, sd, 'random')
                loss = masked_bce_loss(net(x, m), Y[b].to(dev))
                opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss)
        for k, w in ((9, None), (5, FIVE)):
            p = predict(net, vai, mu, sd, w); g = predict(net, gidx, mu, sd, w)
            ix = [pos[u] for u in va]
            (oof9 if k == 9 else oof5)[ix] = p
            (gold9 if k == 9 else gold5)[:] += g / 5
        ok = ~np.isnan(T[[pos[u] for u in va]]).all(1)
        print(f'{arm} fold {f}: last loss {tot / steps:.4f} | val macro(9) {macro_auc(T[[pos[u] for u in va]], oof9[[pos[u] for u in va]]):.4f} | {time.time() - t0:.0f}s', flush=True)
    for name, arr in (('oof9', oof9), ('oof5', oof5), ('gold9', gold9), ('gold5', gold5)):
        np.save(OUT / f'{arm}_{name}.npy', arr)
    res[arm] = dict(oof9=oof9, oof5=oof5, gold9=gold9, gold5=gold5)

# ---- readout --------------------------------------------------------------------------------------------
np.save(OUT / 'truth.npy', T); np.save(OUT / 'gold_true.npy', G)
(OUT / 'labeled_uids.txt').write_text('\n'.join(labeled)); (OUT / 'gold_uids.txt').write_text('\n'.join(gold_uids))
print('\n==== READOUT (pooled OOF vs the fixed truth, steven_v4 binarised) ====')
for arm, r in res.items():
    print(f'{arm:10s} OOF 9 windows {macro_auc(T, r["oof9"]):.4f} | 5 windows {macro_auc(T, r["oof5"]):.4f} | '
          f'gold 9 {macro_auc(G, r["gold9"]):.4f} | 5 {macro_auc(G, r["gold5"]):.4f}')
best = max(res, key=lambda a: macro_auc(T, res[a]['oof9']))
print('arm read by the gate (higher pooled OOF at nine windows):', best)
print('per label OOF 9:', ' '.join(f'{a:.3f}' for a in per_label_auc(T, res[best]['oof9'])))
