"""Assemble the corpus label table the arm c/d screens read.

usage (workspace root): python3 ai/scratch/phase11-claude-labels/assemble.py
Claude labels chunks 0000-0018 (out_corpus/pass1/chunk_NNNN_{a,b}.jsonl); muse-spark labels 0019-0036
(out_corpus/muse_spark/chunk_NNNN.jsonl). Writes ai/scratch/phase11-claude-labels/claude_labels/claude_labels.csv
(uid + 12 labels + 18 atom:* columns, corpus rows only) and a manifest of which labeller wrote which study.
Fails if a chunk is incomplete, a gold study appears, or any cell is NaN: a half-labelled corpus must not ship.
"""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, 'source/src')
from knee.claude_labels import load_label_outputs

D = 'ai/scratch/phase11-claude-labels'
CLAUDE_LAST = 18
frames, source = [], {}
gold = set(pd.read_csv('source/results/gold_study_uids.csv')['StudyInstanceUID'].astype(str))
for chunk in sorted(glob.glob(f'{D}/chunks_corpus/chunk_*.jsonl')):
    stem = os.path.basename(chunk)[:-6]
    uids = [json.loads(line)['uid'] for line in open(chunk, encoding='utf-8')]
    if int(stem[-4:]) <= CLAUDE_LAST:
        files, who = sorted(glob.glob(f'{D}/out_corpus/pass1/{stem}_*.jsonl')), 'claude-sonnet-5-5'
    else:
        files, who = [f'{D}/out_corpus/muse_spark/{stem}.jsonl'], 'opencode/muse-spark-1.3-contributor-free'
    frame = load_label_outputs([f for f in files if os.path.exists(f)], uids)
    if frame.isna().any().any():
        sys.exit(f'{stem}: {int(frame.isna().any(axis=1).sum())} studies missing or with unparsed cells ({who})')
    frames.append(frame)
    source.update({u: who for u in uids})
table = pd.concat(frames)
assert len(table) == 4349 and table.index.is_unique, len(table)
assert not set(table.index) & gold, 'gold studies in the corpus labels'
os.makedirs(f'{D}/claude_labels', exist_ok=True)
table.rename_axis('StudyInstanceUID').reset_index().to_csv(f'{D}/claude_labels/claude_labels.csv', index=False)
pd.Series(source, name='labeller').rename_axis('StudyInstanceUID').reset_index().to_csv(
    f'{D}/claude_labels/labeller_manifest.csv', index=False)
print(table.shape, table.iloc[:, :12].mean().round(3).to_dict())
