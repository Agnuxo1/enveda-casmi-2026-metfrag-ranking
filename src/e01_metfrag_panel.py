# E01 - MetFrag (ipb-halle MetFragRelaunched v2.6.11, LGPL-2.1) FragmenterScore on the public labelled panel
# (sanpier/catalogue-experiment-saner, CC-licensed public Kaggle dataset). Experiment notebook, not a submission.
# Hypothesis H1: MetFrag in-silico fragmentation evidence separates the true structure from the other final-25
# candidates well enough to act as an independent voter in the leader-safe rank fusion.
import os, sys, glob, json, subprocess, shutil, time, hashlib, urllib.request
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

W = Path('/kaggle/working'); T = Path('/kaggle/temp/mf'); T.mkdir(parents=True, exist_ok=True)
env = {}
env['java_preinstalled'] = shutil.which('java')
print('java preinstalled:', env['java_preinstalled'])
try:
    print(subprocess.run(['java', '-version'], capture_output=True, text=True).stderr)
except Exception as e:
    print('java -version failed', e)
JAVA = shutil.which('java')
if JAVA is None:
    url = 'https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jre/hotspot/normal/eclipse'
    urllib.request.urlretrieve(url, T / 'jre.tgz'); subprocess.run(['tar', 'xzf', str(T / 'jre.tgz'), '-C', str(T)], check=True)
    JAVA = glob.glob(str(T / 'jdk-17*-jre/bin/java'))[0]
env['java'] = JAVA
JAR = T / 'MetFragCommandLine-2.6.11.jar'
urllib.request.urlretrieve('https://github.com/ipb-halle/MetFragRelaunched/releases/download/v2.6.11/MetFragCommandLine-2.6.11.jar', JAR)
env['jar_sha256'] = hashlib.sha256(JAR.read_bytes()).hexdigest()
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'rdkit==2026.3.3'], check=False)
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Descriptors, rdMolDescriptors
import rdkit; env['rdkit'] = rdkit.__version__

def find(name):
    hits = glob.glob(f'/kaggle/input/**/{name}', recursive=True); assert hits, name; return hits[0]
cand = pd.read_parquet(find('labelled_candidate_scores.parquet'))
spec = pd.read_parquet(glob.glob('/kaggle/input/**/catalogue_replay_checkpoint/panel/test.parquet', recursive=True)[0])
PROTON = 1.00727646688
SHIFT = {'[M+H]+': (PROTON, 1), '[M+NH4]+': (18.03382555, 18), '[M+Na]+': (22.98922070, 23), '[M+K]+': (38.96315851, 39),
         '[M-H]-': (-PROTON, -1), '[M+Cl]-': (34.96940126, 35), '[M+CH2O2-H]-': (44.99820285, 45),
         '[M-H2O+H]+': (PROTON - 18.01056468, 1)}
spec = spec[spec.adduct.isin(SHIFT)].copy()
spec['tims'] = spec.instrument_type.eq('timsTOF').astype(int)
spec = spec.sort_values(['molecule_id', 'tims', 'num_peaks'], ascending=[True, False, False]).groupby('molecule_id').head(6)
print('spectra used', len(spec), 'molecules', spec.molecule_id.nunique())

def db_rows(g):
    rows = []
    for _, r in g.iterrows():
        m = Chem.MolFromSmiles(r.smiles)
        if m is None: continue
        ik = Chem.MolToInchiKey(m)
        rows.append(dict(Identifier=f'c{int(r.baseline_rank)}', MonoisotopicMass=Descriptors.ExactMolWt(m),
                         MolecularFormula=rdMolDescriptors.CalcMolFormula(m), SMILES=Chem.MolToSmiles(m),
                         InChI=Chem.MolToInchi(m), InChIKey1=ik[:14], InChIKey2=ik[15:25]))
    return pd.DataFrame(rows)

jobs = []
for mid, g in cand.groupby('molecule_id'):
    d = T / mid; d.mkdir(exist_ok=True)
    db_rows(g).to_csv(d / 'db.csv', index=False)
    for _, s in spec[spec.molecule_id == mid].iterrows():
        shift, mode = SHIFT[s.adduct]
        neutral = float(s.precursor_mz) - shift
        sid = str(s.spectrum_id)
        mz = np.asarray(s.ms2_mzs, float); it = np.asarray(s.ms2_normalized_intensities, float)
        keep = mz < float(s.precursor_mz) + 1.0
        pk = d / f'{sid}.txt'
        np.savetxt(pk, np.c_[mz[keep], 999 * it[keep] / max(it[keep].max(), 1e-9)], fmt='%.5f')
        par = d / f'{sid}.par'
        par.write_text('\n'.join([
            f'PeakListPath = {pk}', 'MetFragDatabaseType = LocalCSV', f'LocalDatabasePath = {d / "db.csv"}',
            f'NeutralPrecursorMass = {neutral:.6f}', 'DatabaseSearchRelativeMassDeviation = 50',
            'FragmentPeakMatchAbsoluteMassDeviation = 0.003', 'FragmentPeakMatchRelativeMassDeviation = 10',
            f'PrecursorIonMode = {mode}', f'IsPositiveIonMode = {"True" if mode > 0 else "False"}',
            'MetFragScoreTypes = FragmenterScore', 'MetFragScoreWeights = 1.0', 'MetFragCandidateWriter = CSV',
            f'SampleName = {sid}', f'ResultsPath = {d}', 'MaximumTreeDepth = 2',
            'MetFragPreProcessingCandidateFilter = UnconnectedCompoundFilter', 'NumberThreads = 1']) + '\n')
        jobs.append((mid, sid, s.instrument_type, s.adduct, str(par), str(d / f'{sid}.csv')))
print('jobs', len(jobs))

def run(job):
    mid, sid, inst, add, par, out = job
    t0 = time.time()
    p = subprocess.run([JAVA, '-Xmx1500m', '-jar', str(JAR), par], capture_output=True, text=True, timeout=600)
    if not os.path.exists(out): return [dict(molecule_id=mid, spectrum_id=sid, error=p.stderr[-300:] + p.stdout[-300:])]
    r = pd.read_csv(out)
    return [dict(molecule_id=mid, spectrum_id=sid, instrument=inst, adduct=add, cand=row['Identifier'],
                 fragmenter=float(row['FragmenterScore']), explained=str(row.get('ExplPeaks', '')),
                 n_peaks_used=int(row.get('NumberPeaksUsed', -1)), sec=time.time() - t0)
            for row in r.to_dict('records')]

t0 = time.time(); res = []
with ProcessPoolExecutor(4) as ex:
    for i, rows in enumerate(ex.map(run, jobs)):
        res.extend(rows)
        if i % 100 == 0: print(i, round(time.time() - t0), flush=True)
df = pd.DataFrame(res); df.to_parquet(W / 'e01_metfrag_scores.parquet', index=False)
env['seconds'] = time.time() - t0; env['jobs'] = len(jobs); env['errors'] = int(df.get('error', pd.Series(dtype=str)).notna().sum())
json.dump(env, open(W / 'e01_env.json', 'w'), indent=1); print(env)
print(df.head())
