"""Score the new public label sets: gold AUC (n=58, where the set covers gold), contamination, and
agreement with the current training target (the three-source blend) on the studies both cover."""
import glob, sys, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from knee.metrics import macro_auc, per_label_auc
LABELS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
D = '../ai/scratch/phase9-labels/'
P6 = '../ai/scratch/phase6-labels/'
train = pd.read_csv('data/sample/train.csv')
gold = train[train[LABELS].notna().any(axis=1)].set_index('StudyInstanceUID')
y = gold[LABELS].to_numpy(float)

def conf_fold(v, c): return v * c + (1 - v) * (1 - c)
def load_plain(path):
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    if 'StudyInstanceUID' not in df.columns: return None
    df['StudyInstanceUID'] = df['StudyInstanceUID'].astype(str)
    df = df.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
    if all(l in df.columns for l in LABELS): return df[LABELS].apply(pd.to_numeric, errors='coerce')
    if all(f'{l}__label' in df.columns for l in LABELS):   # laymond
        out = pd.DataFrame(index=df.index)
        for l in LABELS:
            v = pd.to_numeric(df[f'{l}__label'], errors='coerce'); c = pd.to_numeric(df[f'{l}__confidence'], errors='coerce').clip(0,1).fillna(0.5)
            out[l] = conf_fold(v, c)
        return out
    return None

# the current target: probability mean of steven_v4, ours, lixin73 (conf-folded), as in blend_sources
def conf_src(path):
    df = pd.read_csv(path).set_index('StudyInstanceUID'); out = pd.DataFrame(index=df.index)
    for l in LABELS:
        v = pd.to_numeric(df[l], errors='coerce'); cc = f'{l}__conf'
        if cc in df.columns: v = conf_fold(v, pd.to_numeric(df[cc], errors='coerce').clip(0,1).fillna(0.5))
        out[l] = v
    return out
ours = pd.read_csv('results/pseudo_labels_qwen3_4b.csv').set_index('StudyInstanceUID'); ours.columns = [c.replace('score_','') for c in ours.columns]
st = conf_src(P6+'stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv')
lx = conf_src(P6+'lixin73_rsna-knee-llm-report-labels-sol56/report_labels_gpt56sol.csv')
idx = ours.index.intersection(st.index).intersection(lx.index)
target = (ours.loc[idx, LABELS] + st.loc[idx, LABELS] + lx.loc[idx, LABELS]) / 3
tbin = (target >= 0.5).astype(float)

files = [f for f in sorted(glob.glob(D + '*/*.csv') + glob.glob(D + '*/*.parquet'))
         if '/_m/' not in f and not any(k in f for k in ('gold_benchmark','gold74','train_folds','LLM_reasoning','provenance','weights','shard'))]
rows = []
for f in files:
    try: df = load_plain(f)
    except Exception as e: continue
    if df is None: continue
    name = '/'.join(f.split('/')[-2:])[:70]
    g = df.reindex(gold.index)
    cover = g.notna().all(axis=1).mean()
    gm = np.nan; contam = np.nan
    if cover > 0.9:
        s = g.fillna(0.5).to_numpy(float); gm = macro_auc(y, s)
        contam = (((s >= 0.5).astype(float) == y).mean())
    common = df.index.intersection(idx)
    am = np.nan
    if len(common) > 500:
        a = df.loc[common, LABELS].fillna(0.5).to_numpy(float); t = tbin.loc[common].to_numpy(float)
        ok = [l for l in range(12) if 0 < t[:, l].sum() < len(t)]
        am = macro_auc(t[:, ok], a[:, ok])
    rows.append((name, len(df), cover, gm, contam, am))
res = pd.DataFrame(rows, columns=['file','rows','gold_cover','gold_macro','gold_agree','auc_vs_target'])
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 75)
print(res.sort_values('gold_macro', ascending=False).to_string(index=False, float_format=lambda v: f'{v:.4f}'))
