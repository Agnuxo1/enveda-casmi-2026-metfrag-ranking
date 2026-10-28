"""E01b: does rule D keep its effect with fewer spectra per molecule (runtime budget)? Re-uses E01 per-spectrum scores."""
import os, sys, json, numpy as np, pandas as pd, subprocess
mf = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else 'e01_metfrag_scores.parquet')
spec = pd.read_parquet(os.environ.get('CASMI_PANEL_DIR', 'data/catalogue') + r'/catalogue_replay_checkpoint/panel/test.parquet', columns=['spectrum_id', 'num_peaks', 'instrument_type'])
spec['spectrum_id'] = spec.spectrum_id.astype(str); mf['spectrum_id'] = mf.spectrum_id.astype(str)
mf = mf.merge(spec, on='spectrum_id', how='left')
mf['tims'] = mf.instrument_type.eq('timsTOF').astype(int)
order = mf.drop_duplicates('spectrum_id').sort_values(['molecule_id', 'tims', 'num_peaks'], ascending=[True, False, False])
order['k'] = order.groupby('molecule_id').cumcount() + 1
mf = mf.merge(order[['spectrum_id', 'k']], on='spectrum_id')
print('timing by peaks quartile (s/job):', mf.drop_duplicates('spectrum_id').groupby(pd.qcut(mf.drop_duplicates('spectrum_id').num_peaks, 4)).sec.mean().round(1).to_dict())
res = {}
for K in [1, 2, 3, 6]:
    sub = mf[mf.k <= K].drop(columns=['k']); p = f'_e01_k{K}.parquet'; sub.to_parquet(p)
    out = json.loads(subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'e01_eval.py'), p], capture_output=True, text=True).stdout)
    res[K] = {key: out[key] for key in ['standalone_metfrag', 'D_c472pos+mf0.25', 'D_c472pos+mf0.5', 'D_c472pos+mf0.75']}
    res[K]['jobs'] = int(sub.spectrum_id.nunique())
print(json.dumps(res, indent=1)); json.dump(res, open('e01b_spectra_budget.json', 'w'), indent=1)
