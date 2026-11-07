"""Blend kernel: compliant public stack (P) + our own ensemble, per-label rank mean.

usage: build_blend.py <P notebook dir> <our submit notebook dir> <out dir> [w_ours]

Our notebook's code cells run unchanged as a subprocess (own namespace, own cwd) so nothing in the
public pipeline's globals can collide with ours. If ours fails the blend degrades to P alone and
says so in the log; if P fails the notebook fails (P's own gates are fatal by design).
"""
import glob, json, os, sys

P_DIR, OURS_DIR, OUT = sys.argv[1:4]
W_OURS = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
DEADLINE_H = float(sys.argv[5]) if len(sys.argv) > 5 else 7.9  # of a 9 h limit, less Kaggle's session start-up and a margin
assert 0.0 < W_OURS < 1.0

p_nb = json.load(open(glob.glob(f'{P_DIR}/*.ipynb')[0]))
o_nb = json.load(open(glob.glob(f'{OURS_DIR}/*.ipynb')[0]))
o_meta = json.load(open(f'{OURS_DIR}/kernel-metadata.json'))
p_meta = json.load(open(f'{P_DIR}/kernel-metadata.json'))

ours_script = '\n\n'.join(
    ''.join(l for l in c['source'] if not l.lstrip().startswith(('!', '%')))
    for c in o_nb['cells'] if c['cell_type'] == 'code')
assert "submission.csv" in ours_script and 'build_submission' in ours_script
compile(ours_script, 'ours.py', 'exec')

run_ours = f'''# Our ensemble, as its own process: nothing it defines can collide with the public pipeline.
# It gets a hard deadline: the hidden-set rerun of the public stack is slow (a large cohort runs its
# CoAtNet readers one at a time), so if ours cannot finish inside the session limit it is killed and
# the public stack's own submission.csv goes out alone instead of the whole notebook timing out.
import os, subprocess, sys, time
OURS_WORKDIR = '/tmp/ours'
os.makedirs(OURS_WORKDIR, exist_ok=True)
with open(f'{{OURS_WORKDIR}}/ours.py', 'w') as _f:
    _f.write({ours_script!r})
BLEND_DEADLINE = BLEND_T0 + {DEADLINE_H} * 3600
_remaining = BLEND_DEADLINE - time.time()
print('public stack finished after %.2f h; ours may use %.2f h' % ((time.time() - BLEND_T0) / 3600, max(_remaining, 0) / 3600))
if _remaining < 900:
    OURS_RC = -2
    print('too little time left for our ensemble; skipping it')
else:
    try:
        OURS_RC = subprocess.run([sys.executable, '-u', 'ours.py'], cwd=OURS_WORKDIR, timeout=_remaining).returncode
    except subprocess.TimeoutExpired:
        OURS_RC = -1
        print('our ensemble hit its deadline and was killed')
print('our ensemble exit code:', OURS_RC, '| total %.2f h' % ((time.time() - BLEND_T0) / 3600))
'''

blend = f'''# Per-label rank mean of the two stacks. Equal weight by design: the public stack's weights saw our
# validation studies, so there is no OOF for it on our folds and any fitted weight would be tuned on
# the ~390 public studies.
import numpy as np, pandas as pd
W_OURS = {W_OURS}
UID = 'StudyInstanceUID'
pub = pd.read_csv('/kaggle/working/submission.csv', dtype={{UID: str}})
labels = [c for c in pub.columns if c != UID]
assert len(labels) == 12, labels
ours_path = '/tmp/ours/submission.csv'
if OURS_RC == 0 and os.path.exists(ours_path):
    ours = pd.read_csv(ours_path, dtype={{UID: str}})
    assert list(ours.columns) == list(pub.columns), 'header differs between the two stacks'
    assert set(ours[UID]) == set(pub[UID]) and ours[UID].is_unique, 'study sets differ'
    ours = ours.set_index(UID).loc[pub[UID]].reset_index()
    out = pub.copy()
    out[labels] = ((1 - W_OURS) * pub[labels].rank(pct=True, method='average').to_numpy()
                   + W_OURS * ours[labels].rank(pct=True, method='average').to_numpy())
    print('blended: public stack', 1 - W_OURS, '+ ours', W_OURS)
    corr = [float(np.corrcoef(pub[l], ours[l])[0, 1]) for l in labels]
    print('per-label correlation between stacks: min %.3f max %.3f' % (min(corr), max(corr)))
else:
    out = pub
    print('WARNING: OUR ENSEMBLE FAILED (exit code %s); submitting the public stack alone' % OURS_RC)
assert out[labels].notna().all().all() and np.isfinite(out[labels].to_numpy()).all()
assert ((out[labels] >= 0) & (out[labels] <= 1)).all().all()
out.to_csv('/kaggle/working/submission.csv', index=False)
print('FINAL', len(out), 'studies written')
'''

def cell(src):
    return dict(cell_type='code', metadata={}, source=src.splitlines(keepends=True), outputs=[], execution_count=None)

p_nb['cells'] = [cell('import time\nBLEND_T0 = time.time()\n')] + p_nb['cells'] + [cell(run_ours), cell(blend)]
os.makedirs(OUT, exist_ok=True)
json.dump(p_nb, open(f'{OUT}/knee-phase9-submit-blend.ipynb', 'w'), indent=1)

meta = dict(p_meta)
meta['id'] = 'mohammadyaserhussain/knee-phase9-submit-blend'
meta['title'] = 'knee-phase9-submit-blend'
meta['code_file'] = 'knee-phase9-submit-blend.ipynb'
meta['dataset_sources'] = sorted(set(p_meta['dataset_sources']) | set(o_meta['dataset_sources']))
# the public notebook may itself read kernel outputs (the full stack mounts two), so keep both lists
meta['kernel_sources'] = sorted(set(p_meta.get('kernel_sources') or []) | set(o_meta['kernel_sources']))
json.dump(meta, open(f'{OUT}/kernel-metadata.json', 'w'), indent=1)
print('wrote', OUT, '| datasets', len(meta['dataset_sources']), '| kernel_sources', len(meta['kernel_sources']))
