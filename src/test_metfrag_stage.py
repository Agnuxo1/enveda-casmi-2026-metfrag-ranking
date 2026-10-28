import os
import time, json, tempfile, shutil, numpy as np, pandas as pd
from pathlib import Path
src = open(os.path.join(os.path.dirname(__file__), 'metfrag_stage.py'), encoding='utf-8').read(); src = src[:src.index('\ntry:\n    MF_OK')]
tmp = Path(tempfile.mkdtemp()); comp = tmp / 'comp'; comp.mkdir()
spec = pd.read_parquet(os.environ.get('CASMI_PANEL_DIR', 'data/catalogue') + r'/catalogue_replay_checkpoint/panel/test.parquet')
cand = pd.read_parquet(os.environ.get('CASMI_PANEL_DIR', 'data/catalogue') + r'/catalogue_experiment/labelled_candidate_scores.parquet')
mols = sorted(cand.molecule_id.unique())[:5]
spec[spec.molecule_id.isin(mols)].to_parquet(comp / 'test.parquet')
rows = [(m, ';'.join(cand[cand.molecule_id == m].sort_values('baseline_rank').smiles)) for m in mols]
rows.append(('m_placeholder', ';'.join(['CCO'] * 25)))
pd.DataFrame(rows, columns=['molecule_id', 'smiles']).to_csv(tmp / 'submission.csv', index=False)
g = dict(ROOT=tmp, COMP=comp, NB_T0=time.time(), SUBMITTED='CUSTOM', VARIANT='CUSTOM', SAMPLE_IDS=[r[0] for r in rows])
exec(src, g)
rng = np.random.default_rng(0)
def fake_job(jar, d, sid, s):
    ids = pd.read_csv(d / 'db.csv').Identifier
    return {i: float(rng.random()) for i in ids}
g['_job'] = fake_job; g['_metfrag_jar'] = lambda: 'fake.jar'
import shutil as sh; g['shutil'].which = lambda x: '/usr/bin/java'
ok = g['run_metfrag_stage'](); print('ok', ok, json.dumps(g['MF_REPORT']))
new = pd.read_csv(tmp / 'submission.csv'); old = pd.read_csv(tmp / 'submission_control_c472.csv')
for a, b in zip(new.smiles, old.smiles):
    a, b = a.split(';'), b.split(';'); assert a[0] == b[0] and sorted(a) == sorted(b)
print('rows changed', int((new.smiles != old.smiles).sum()), 'placeholder untouched', new.smiles.iloc[-1] == old.smiles.iloc[-1])
# budget overrun path: tiny budget -> partial application must still validate
g['MF_BUDGET_S'] = 0.0; shutil.copyfile(tmp / 'submission_control_c472.csv', tmp / 'submission.csv')
print('partial', g['run_metfrag_stage'](), json.dumps(g['MF_REPORT']))
shutil.rmtree(tmp)
