"""Build the licence-compliant public stack (P) from jiweiliu/rsna-knee-fast-2xt4-inference.

Drops Stage 2 (A5, DINOv3-derived weights) and Stage 3 (RadImageNet heads), keeps DINOv2 x20,
Raptor x4 views and the CoAtNet readers. Every edit is an asserted text swap so a changed parent
fails loudly instead of silently building something else.
"""
import copy, glob, json, subprocess, sys, tempfile, os

SRC = sys.argv[1]            # pulled parent dir
OUT = sys.argv[2]            # notebooks/phase9-public-stack
nb = json.load(open(glob.glob(f'{SRC}/*.ipynb')[0]))
cells = nb['cells']
text = lambda i: ''.join(cells[i]['source'])
set_text = lambda i, s: cells[i].__setitem__('source', s.splitlines(keepends=True))

def swap(i, old, new=''):
    s = text(i)
    assert s.count(old) == 1, f'cell {i}: expected exactly one match for {old[:60]!r}, found {s.count(old)}'
    set_text(i, s.replace(old, new))

# cell 4: A5 root preflight
lines = text(4).splitlines(keepends=True)
a = next(n for n, l in enumerate(lines) if l.lstrip().startswith("roots=[Path(p) for p in ['/kaggle/input/datasets/mattiaangeli/knee-mri-fold-weights'"))
b = next(n for n, l in enumerate(lines) if "receipt['A5_root']" in l)
assert 0 < b - a < 8, (a, b)
set_text(4, ''.join(lines[:a] + lines[b + 1:]))
# cell 29: the calibrator belongs to the RadImageNet stage
swap(29, "if not globals().get('V18_CALIBRATOR_APPLIED'):raise RuntimeError('BTKD calibrator was not applied')\n")
swap(29, 'a5_folds=5', 'a5_folds=0')

# stages 2 (A5) and 3 (RadImageNet): cells 18-25 (headers 18 and 24 included)
assert text(18).startswith('## Stage 2') and text(24).startswith('## Stage 3') and text(26).startswith('## Stages 4')
assert 'RadImageNet' in text(24) and '_RAD_CAL_PAYLOAD' in text(25)
keep = [c for i, c in enumerate(cells) if not 18 <= i <= 25]
nb['cells'] = keep

# static check: names the removed cells defined must not be used by what remains
code = '\n\n'.join(''.join(c['source']) for c in keep if c['cell_type'] == 'code')
with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
    f.write(code)
out = subprocess.run([sys.executable, '-m', 'pyflakes', f.name], capture_output=True, text=True).stdout
undefined = [l for l in out.splitlines() if 'undefined name' in l]
print('\n'.join(undefined) or 'pyflakes: no undefined names')
os.unlink(f.name)

os.makedirs(OUT, exist_ok=True)
for c in nb['cells']:
    if c['cell_type'] == 'code':
        c['outputs'] = []; c['execution_count'] = None
json.dump(nb, open(f'{OUT}/knee-phase9-public-stack.ipynb', 'w'), indent=1)

meta = json.load(open(f'{SRC}/kernel-metadata.json'))
drop = ('knee-mri-fold-weights', 'resnet-50-radimagenet', 'rsna-knee-e11-diverse', 'rsna-knee-e9-radimagenet', 'rsna-knee-v52-radimagenet')
meta['dataset_sources'] = [d for d in meta['dataset_sources'] if not any(x in d for x in drop)]
meta['kernel_sources'] = []
meta.update(id='mohammadyaserhussain/knee-phase9-public-stack', title='knee-phase9-public-stack',
            code_file='knee-phase9-public-stack.ipynb', is_private='true', enable_internet='false',
            enable_gpu='true', machine_shape='NvidiaTeslaT4')
for k in ('id_no', 'keywords'):
    meta.pop(k, None)
json.dump(meta, open(f'{OUT}/kernel-metadata.json', 'w'), indent=1)
print('datasets:', meta['dataset_sources'])
