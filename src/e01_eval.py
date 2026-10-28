"""E01 evaluation: MetFrag voter on the labelled panel. Usage: python e01_eval.py <e01_metfrag_scores.parquet>"""
import os, sys, json, numpy as np, pandas as pd
P = os.environ.get('CASMI_PANEL_DIR', 'data/catalogue') + r'/catalogue_experiment/labelled_candidate_scores.parquet'
d = pd.read_parquet(P); mf = pd.read_parquet(sys.argv[1])
mf = mf[mf.get('error').isna()] if 'error' in mf else mf
mf['rank_key'] = mf.cand.str[1:].astype(int)
# per-spectrum max-normalisation, then mean over spectra; candidates outside MetFrag's mass window get 0
mf['norm'] = mf.fragmenter / mf.groupby(['molecule_id', 'spectrum_id']).fragmenter.transform('max').replace(0, np.nan)
agg = mf.groupby(['molecule_id', 'rank_key']).norm.mean().fillna(0).rename('mf').reset_index()
d = d.merge(agg, left_on=['molecule_id', 'baseline_rank'], right_on=['molecule_id', 'rank_key'], how='left')
d['mf'] = d.mf.fillna(0.0)
d['nrank'] = d.groupby('molecule_id').neural_score.rank(ascending=False, method='first')
d['mrank'] = d.groupby('molecule_id').mf.rank(ascending=False, method='average')
G = {m: g.sort_values('baseline_rank').reset_index(drop=True) for m, g in d.groupby('molecule_id')}
mols = sorted(G); covered = [m for m in mols if (G[m].mf > 0).any()]

def order(g, w_top=0.25, w_mid=0.75, w_mf=0.0, mf_top=False, k=3.0):
    b = g.baseline_rank.values.astype(float); n = g.nrank.values; r = g.mrank.values
    s_top = 1/(k+b) + w_top/(k+n) + (w_mf/(k+r) if mf_top else 0)
    s_mid = 1/(k+b) + w_mid/(k+n) + w_mf/(k+r)
    lead = int(np.argmax(s_top)); return [lead] + [i for i in np.argsort(-s_mid, kind='stable') if i != lead]

def rr(g, o):
    hit = np.flatnonzero(g.label.values[o][:25] == 1); return 1/(hit[0]+1) if hit.size else 0.0

def score(fn, ms): return np.array([rr(G[m], fn(G[m])) for m in ms])
rng = np.random.default_rng(20261028); out = {}
def boot(diff): b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(4000)]; return np.percentile(b, [2.5, 97.5])
# standalone discrimination: MRR of the true structure when ordering by each signal alone
for name, col, asc in [('baseline', 'baseline_rank', True), ('neural', 'neural_score', False), ('metfrag', 'mf', False)]:
    s = score(lambda g: list(np.argsort(g[col].values if asc else -g[col].values, kind='stable')), covered)
    out[f'standalone_{name}'] = round(float(s.mean()), 4)
# random-order expectation
out['standalone_random_expect'] = round(float(np.mean([np.mean([1/(i+1) for i in range(len(G[m]))]) for m in covered])), 4)
ctrl = score(lambda g: order(g), mols); out['control_C472'] = round(float(ctrl.mean()), 4)
for w in [0.1, 0.25, 0.5, 0.75]:
    for top in [False, True]:
        s = score(lambda g: order(g, w_mf=w, mf_top=top), mols); diff = s - ctrl; lo, hi = boot(diff)
        out[f'C472+mf{w}{"_top" if top else ""}'] = dict(mrr=round(float(s.mean()), 4), delta=round(float(diff.mean()), 4),
                                                          ci95=[round(lo, 4), round(hi, 4)], moved=int((diff != 0).sum()))
# fold stability: three molecule-disjoint folds (seeded), delta per fold for w_mf=0.25
f = np.random.default_rng(7).permutation(len(mols)) % 3
for w in [0.25, 0.5]:
    s = score(lambda g: order(g, w_mf=w), mols); diff = s - ctrl
    out[f'folds_mf{w}'] = [round(float(diff[f == i].mean()), 4) for i in range(3)]

# deployable rule D: take the C472 order as one voter; keep its leader; ranks 2-25 by 1/(k+pos) + w/(k+metfrag rank)
def order_D(g, w, k=3.0):
    o = order(g); pos = np.empty(len(o)); pos[o] = np.arange(1, len(o) + 1)
    r = g.mrank.values; s = 1/(k+pos) + w/(k+r); lead = o[0]
    return [lead] + [i for i in np.argsort(-s, kind='stable') if i != lead]
for w in [0.1, 0.25, 0.5, 0.75, 1.0]:
    s = score(lambda g: order_D(g, w), mols); diff = s - ctrl; lo, hi = boot(diff)
    out[f'D_c472pos+mf{w}'] = dict(mrr=round(float(s.mean()), 4), delta=round(float(diff.mean()), 4), ci95=[round(lo, 4), round(hi, 4)],
                                   moved=int((diff != 0).sum()), folds=[round(float(diff[f == i].mean()), 4) for i in range(3)])
out['n_mol'] = len(mols); out['n_covered'] = len(covered)
print(json.dumps(out, indent=1)); json.dump(out, open(os.environ.get('E01_OUT', 'e01_eval.json'), 'w'), indent=1)
