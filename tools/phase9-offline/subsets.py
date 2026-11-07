"""Best small member subsets by pooled OOF at nine groups, with a rough cost (336px models weigh 1.6)."""
import itertools, sys, numpy as np
sys.argv = ['x']; exec(open('ai/scratch/phase9-offline/score8.py').read().split("seven = [")[0])
cost = {n: (1.6 if n in ('s2', 'd3_336') else 1.0) for n in M}
names = list(M)
rows = []
for k in (2, 3, 4):
    for sub in itertools.combinations(names, k):
        oof = macro_auc(ytrue, mean(sub, 0)); gold = macro_auc(gtrue, mean(sub, 1))
        rows.append((k, sub, oof, gold, 5 * sum(cost[n] for n in sub)))
for k in (2, 3, 4):
    best = sorted([r for r in rows if r[0] == k], key=lambda r: -r[2])[:4]
    for r in best: print(k, '+'.join(r[1]), f'OOF {r[2]:.4f} gold {r[3]:.4f} cost {r[4]:.0f} model-units')
print('all8', f'OOF {macro_auc(ytrue, mean(names,0)):.4f}', 'cost', 5 * sum(cost.values()))
