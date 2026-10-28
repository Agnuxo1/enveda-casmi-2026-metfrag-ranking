"""Animated figures for the README. Inputs are public: our own submission history (Kaggle API),
the public leaderboard export and the public labelled panel. No competition data is stored in the repo."""
import json, os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch
OUT = 'figures'; C = os.environ.get('CASMI_CACHE', 'data') + '/'
INK, MUTED, GRID, BG = '#1d2433', '#7a8194', '#e6e8ee', '#ffffff'
BLUE, ORANGE, GREEN, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#e34948'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.edgecolor': '#c9ccd6', 'axes.titleweight': 'bold',
                     'axes.titlelocation': 'left', 'figure.facecolor': BG, 'axes.facecolor': BG, 'font.size': 11})

def save(anim, name, fps=12):
    anim.save(f'{OUT}/{name}', writer=PillowWriter(fps=fps)); print('wrote', name)

# 1 - leaderboard: score distribution with our position climbing as submissions land
def leaderboard_journey(extra=None):
    lb = pd.read_csv(C + 'leaderboard_20261028.csv'); sc = lb.Score.astype(float).values
    subs = json.load(open(C + 'submissions_api_20261028.json'))
    hist = [(s['date'][:10], float(s['public_score'])) for s in subs if s['public_score'] not in ('', 'None')][::-1]
    if extra: hist += extra
    best = np.maximum.accumulate([h[1] for h in hist])
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=90)
    bins = np.linspace(0.0, 0.52, 105)
    def frame(i):
        ax.clear(); ax.hist(sc, bins=bins, color='#cfd8e8', edgecolor='white')
        ax.set_xlim(0.2, 0.53); ax.set_yscale('log'); ax.grid(axis='y', color=GRID)
        b = best[min(i, len(best) - 1)]; rank = int((sc > b).sum()) + 1
        ax.axvline(b, color=ORANGE, lw=3); ax.axvline(sc.max(), color=MUTED, ls='--', lw=1)
        ax.text(b + 0.003, ax.get_ylim()[1] * 0.35, f'our best {b:.3f}\n≈ rank {rank:,} / {len(sc):,}', color=ORANGE, weight='bold')
        ax.text(sc.max() - 0.004, ax.get_ylim()[1] * 0.35, f'leader\n{sc.max():.3f}', color=MUTED, ha='right')
        d = hist[min(i, len(hist) - 1)][0]
        ax.set_title(f'Public leaderboard (4,227 teams, 28 Oct 2026) — submission {min(i+1, len(hist))}/{len(hist)} · {d}')
        ax.set_xlabel('public MRR@25'); ax.set_ylabel('teams (log)')
    save(FuncAnimation(fig, frame, frames=len(hist) + 8, interval=350), 'leaderboard_journey.gif', fps=3); plt.close(fig)

# 2 - leader-safe reciprocal-rank fusion on a real panel molecule, w_mid sweeping 0 -> 1
def rrf_animation():
    d = pd.read_parquet(C + 'ds/catalogue/catalogue_experiment/labelled_candidate_scores.parquet')
    d['nrank'] = d.groupby('molecule_id').neural_score.rank(ascending=False, method='first')
    pick = None
    for m, g in d.groupby('molecule_id'):
        tr = g.loc[g.label == 1, 'baseline_rank']
        if len(tr) and 3 <= tr.iloc[0] <= 9 and g.loc[g.label == 1, 'nrank'].iloc[0] <= 3: pick = g.sort_values('baseline_rank').head(12); break
    b = pick.baseline_rank.values.astype(float); n = pick.nrank.values; lab = pick.label.values
    ws = np.r_[np.linspace(0, 1, 40), np.ones(10)]
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=90)
    def order(w):
        st = 1/(3+b) + 0.25/(3+n); sm = 1/(3+b) + w/(3+n); lead = int(np.argmax(st))
        return [lead] + [i for i in np.argsort(-sm, kind='stable') if i != lead]
    def frame(i):
        w = ws[i]; o = order(w); ax.clear(); ax.set_xlim(0, 10); ax.set_ylim(12.6, -0.6); ax.axis('off')
        for pos, idx in enumerate(o):
            col = GREEN if lab[idx] else (BLUE if pos == 0 else '#e9edf5')
            ax.add_patch(FancyBboxPatch((0.3, pos - 0.38), 9.2, 0.76, boxstyle='round,pad=0.02,rounding_size=0.18', color=col))
            tc = 'white' if (lab[idx] or pos == 0) else INK
            ax.text(0.6, pos, f'#{pos+1}', va='center', color=tc, weight='bold')
            ax.text(1.6, pos, f'pipeline rank {int(b[idx]):>2}   ·   neural rank {int(n[idx]):>2}' + ('   ← correct structure' if lab[idx] else ''), va='center', color=tc, family='DejaVu Sans Mono', fontsize=10)
        tr = o.index(int(np.flatnonzero(lab)[0])) + 1
        ax.set_title(f'Leader-safe RRF · rank 1 fixed, ranks 2–25 neural weight w = {w:.2f}\nreciprocal rank of the correct structure: {1/tr:.2f}', fontsize=11.5)
    save(FuncAnimation(fig, frame, frames=len(ws), interval=120), 'rrf_leader_safe.gif', fps=10); plt.close(fig)

# 3 - pipeline flow
def pipeline_flow(stages):
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=90)
    def frame(i):
        ax.clear(); ax.set_xlim(0, len(stages)); ax.set_ylim(0, 1); ax.axis('off')
        for j, (t, s, c) in enumerate(stages):
            on = j <= i // 4
            ax.add_patch(FancyBboxPatch((j + 0.06, 0.22), 0.88, 0.56, boxstyle='round,pad=0.02,rounding_size=0.06', color=c if on else '#eef0f4'))
            ax.text(j + 0.5, 0.6, t, ha='center', va='center', weight='bold', color='white' if on else MUTED, fontsize=10.5)
            ax.text(j + 0.5, 0.38, s, ha='center', va='center', color='white' if on else MUTED, fontsize=8.3)
            if j < len(stages) - 1: ax.annotate('', (j + 1.06, 0.5), (j + 0.94, 0.5), arrowprops=dict(arrowstyle='->', color=INK if on else GRID, lw=2))
        ax.set_title('From MS/MS spectra to a ranked list of 25 structures', fontsize=12)
    save(FuncAnimation(fig, frame, frames=4 * len(stages) + 10, interval=150), 'pipeline_flow.gif', fps=6); plt.close(fig)

if __name__ == '__main__':
    leaderboard_journey(); rrf_animation()
    pipeline_flow([('MS/MS input', 'timsTOF spectra\nadducts · CE', '#4a5568'),
                   ('Candidates', 'library + COCONUT\nChEBI · LIPID MAPS\nPubChem 105.9 M', BLUE),
                   ('Engine scores', 'fingerprint nets\nGLACIER · ICEBERG', VIOLET),
                   ('0.465 fusion', 'identity-aware\nRRF + PubChem gate', ORANGE),
                   ('Neural RRF', 'spectrum↔structure\nleader-safe w=0.75', GREEN),
                   ('MetFrag voter', 'in-silico\nfragmentation (E01)', RED),
                   ('MRR@25', 'InChIKey14\nRDKit 2026.03.3', '#4a5568')])


# 4 - MetFrag evidence: one panel spectrum, candidates cycling, explained peaks highlighted
def metfrag_evidence():
    mf = pd.read_parquet(C + 'e01_metfrag_scores.parquet')
    lab = pd.read_parquet(C + 'ds/catalogue/catalogue_experiment/labelled_candidate_scores.parquet')
    lab['cand'] = 'c' + lab.baseline_rank.astype(str)
    mf = mf.merge(lab[['molecule_id', 'cand', 'label', 'baseline_rank']], on=['molecule_id', 'cand'])
    best = None
    for (m, s), g in mf.groupby(['molecule_id', 'spectrum_id']):
        g = g.sort_values('fragmenter', ascending=False)
        if len(g) >= 8 and g.label.iloc[0] == 1 and g.baseline_rank.iloc[0] > 1 and 30 <= g.n_peaks_used.iloc[0] <= 160:
            best = g.head(6); break
    if best is None:
        best = mf[mf.label == 1].sort_values('fragmenter', ascending=False).head(1)
    def peaks(txt):
        if not isinstance(txt, str) or not txt or txt == 'nan': return np.zeros((0, 2))
        return np.array([[float(x) for x in p.split('_')[:2]] for p in txt.split(';') if '_' in p])
    allp = np.vstack([peaks(t) for t in best.explained]); mzmax = allp[:, 0].max() * 1.05 if len(allp) else 500
    top = best.fragmenter.max()
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=90)
    seq = [i for i in range(len(best)) for _ in range(10)]
    def frame(t):
        i = seq[t]; r = best.iloc[i]; p = peaks(r.explained); ax.clear()
        ax.vlines(allp[:, 0], 0, allp[:, 1], color='#d5d9e2', lw=1.2)
        col = GREEN if r.label == 1 else ORANGE
        if len(p): ax.vlines(p[:, 0], 0, p[:, 1], color=col, lw=2.2)
        ax.set_xlim(40, mzmax); ax.set_ylim(0, 1080); ax.set_xlabel('m/z'); ax.set_ylabel('relative intensity')
        tag = 'correct structure' if r.label == 1 else 'decoy candidate'
        ax.set_title(f'MetFrag: peaks explained by candidate (pipeline rank {int(r.baseline_rank)}, {tag})', fontsize=11.5)
        ax.text(0.99, 0.93, f'FragmenterScore {r.fragmenter / top:.2f} (relative)\nexplained peaks {len(p)}', transform=ax.transAxes, ha='right', va='top', color=col, weight='bold')
    save(FuncAnimation(fig, frame, frames=len(seq), interval=160), 'metfrag_evidence.gif', fps=5); plt.close(fig)


# 5 - E01: signal strength and fusion effect, bars growing
def e01_results():
    e = json.load(open('experiments/e01_eval_k6.json'))
    names = ['random order', 'neural model', 'MetFrag 2.6.11']; vals = [e['standalone_random_expect'], e['standalone_neural'], e['standalone_metfrag']]
    cols = ['#b8bfcc', VIOLET, RED]
    ws = ['0.1', '0.25', '0.5', '0.75', '1.0']; ds = [e[f'D_c472pos+mf{w}'] for w in ws]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=90, gridspec_kw=dict(width_ratios=[1, 1.25]))
    N = 30
    def frame(t):
        f = min(1, t / N); a1.clear(); a2.clear()
        a1.barh(names, [v * f for v in vals], color=cols); a1.set_xlim(0, 0.65); a1.invert_yaxis()
        for i, v in enumerate(vals): a1.text(v * f + 0.01, i, f'{v * f:.3f}', va='center', weight='bold')
        a1.set_title('Which candidate is right? MRR using one signal alone', fontsize=11); a1.set_xlabel('MRR among the final 25 (panel, 118 molecules)')
        g = min(1, max(0, (t - N) / N))
        for i, d in enumerate(ds):
            lo, hi = d['ci95']; a2.plot([lo * g, hi * g], [i, i], color='#9aa3b5', lw=3)
            a2.plot(d['delta'] * g, i, 'o', color=GREEN if lo > 0 else BLUE, ms=10)
        a2.axvline(0, color=INK, lw=1); a2.set_yticks(range(len(ws))); a2.set_yticklabels([f'w = {w}' for w in ws]); a2.invert_yaxis()
        a2.set_xlim(-0.006, 0.013); a2.set_xlabel('MRR gain vs 0.472 control (paired bootstrap 95% CI)')
        a2.set_title('MetFrag as a voter on ranks 2–25 (rank 1 fixed)', fontsize=11)
    save(FuncAnimation(fig, frame, frames=2 * N + 15, interval=60), 'e01_metfrag_results.gif', fps=15); plt.close(fig)


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'metfrag':
    metfrag_evidence(); e01_results()
