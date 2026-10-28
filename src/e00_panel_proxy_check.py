"""Offline harness: re-ranking variants on the public labelled panel (sanpier/catalogue-experiment-saner).
Panel = 120 training-library molecules, final-25 lists of the pinned 0.465 pipeline with neural scores.
Metric = MRR@25 on metric_key (InChIKey14 after official tautomer standardisation, precomputed in the panel)."""
import os, numpy as np, pandas as pd, sys, json
P = os.environ.get('CASMI_PANEL_DIR', 'data/catalogue') + r'/catalogue_experiment/labelled_candidate_scores.parquet'
d = pd.read_parquet(P)
d['nrank'] = d.groupby('molecule_id')['neural_score'].rank(ascending=False, method='first')

def consensus(g, w_top, w_mid, k=3.0, kn=None):
    kn = k if kn is None else kn
    b = g['baseline_rank'].values.astype(float); n = g['nrank'].values
    s_top = 1/(k+b) + w_top/(kn+n); s_mid = 1/(k+b) + w_mid/(kn+n)
    lead = int(np.argmax(s_top)); rest = [i for i in np.argsort(-s_mid, kind='stable') if i != lead]
    return [lead] + rest

def mrr(order_fn, mols):
    out = []
    for m in mols:
        g = groups[m]; order = order_fn(g)
        lab = g['label'].values[order]
        hit = np.flatnonzero(lab[:25] == 1)
        out.append(1/(hit[0]+1) if hit.size else 0.0)
    return np.array(out)

groups = {m: g.sort_values('baseline_rank').reset_index(drop=True) for m, g in d.groupby('molecule_id')}
mols = sorted(groups)
variants = {'V0_baseline': (0, 0), 'V1_0.25/0.25': (0.25, 0.25), 'V2_0.25/0.50': (0.25, 0.5),
            'C472_0.25/0.75': (0.25, 0.75), 'V4_0.25/1.00': (0.25, 1.0)}
res = {name: mrr(lambda g, w=w: consensus(g, *w), mols) for name, w in variants.items()}
rng = np.random.default_rng(0)
base = res['V1_0.25/0.25']
print('n molecules', len(mols), 'molecules not at rank1 under V0:', int((res['V0_baseline'] < 1).sum()))
for name, r in res.items():
    diff = r - base
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    print(f'{name:16s} MRR={r.mean():.4f}  dV1={diff.mean():+.4f}  CI95=[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]  moved={int((diff!=0).sum())}')
