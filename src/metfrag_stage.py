# ── Stage 4 · MetFrag voter on ranks 2–25 (Enveda CASMI C01, Francisco Angulo de Lafuente) ─────────────
# MetFrag 2.6.11 (IPB Halle, LGPL-2.1; Ruttkies et al. 2016, doi:10.1186/s13321-016-0115-9) scores how well the
# in-silico fragments of each candidate explain the measured MS/MS peaks. Rule D (validated offline as E01):
# the control order is one voter, MetFrag rank the other; rank 1 never changes and no candidate is added or removed.
# Any failure, timeout or validation error keeps the control file untouched.
import collections, glob, json, os, shutil, subprocess, time, hashlib, re, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd

MF_WEIGHT = 0.25         # w in 1/(3 + control position) + w/(3 + MetFrag rank)
MF_SPECTRA = 6           # spectra per molecule (timsTOF first, most peaks first)
MF_BUDGET_S = 90 * 60    # wall-clock budget for the whole stage
MF_DEADLINE_H = 7.2      # never start the stage after this many hours of session time
MF_STOP_H = 8.6          # hard stop for MetFrag jobs (session limit is 9 h)
MF_ROOT = ROOT / 'metfrag'; MF_ROOT.mkdir(exist_ok=True)
MF_REPORT = {'weight': MF_WEIGHT, 'status': 'not started'}
MF_JAVA = None
_PROTON = 1.00727646688
_SHIFT = {'[M+H]+': (_PROTON, 1), '[M+NH4]+': (18.03382555, 18), '[M+Na]+': (22.98922070, 23),
          '[M+K]+': (38.96315851, 39), '[M-H]-': (-_PROTON, -1), '[M+Cl]-': (34.96940126, 35),
          '[M+CH2O2-H]-': (44.99820285, 45), '[M-H2O+H]+': (_PROTON - 18.01056468, 1)}


def _metfrag_jar():
    hits = glob.glob('/kaggle/input/**/MetFragCommandLine-2.6.11.jar', recursive=True)
    return hits[0] if hits else None


def _runtime_preflight(jar):
    """Reject an incompatible JVM before opening spectra or launching jobs."""
    java = os.environ.get('METFRAG_JAVA') or shutil.which('java')
    if not java:
        return None, {'status': 'java missing'}
    try:
        version = subprocess.run([java, '-version'], capture_output=True, text=True, timeout=10)
        match = re.search(r'version\s+"(\d+)(?:\.(\d+))?', version.stderr + version.stdout)
        if version.returncode or not match:
            return None, {'status': 'java version unavailable'}
        major = int(match[1]); major = int(match[2]) if major == 1 else major
        with zipfile.ZipFile(jar) as z:
            manifest = z.read('META-INF/MANIFEST.MF').decode('utf-8')
            main = re.search(r'^Main-Class:\s*(\S+)', manifest, re.MULTILINE)
            if not main:
                return None, {'status': 'jar main class missing'}
            with z.open(main[1].replace('.', '/') + '.class') as f:
                header = f.read(8)
        if header[:4] != b'\xca\xfe\xba\xbe' or len(header) != 8:
            return None, {'status': 'invalid class header'}
        required = int.from_bytes(header[6:8], 'big') - 44
        info = {'status': 'compatible' if major >= required else 'incompatible Java version',
                'java_major': major, 'required_java_major': required}
        return (java if major >= required else None), info
    except (OSError, subprocess.SubprocessError, zipfile.BadZipFile, KeyError, UnicodeError):
        return None, {'status': 'runtime preflight failed'}


def _db(smiles, path):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDLogger.DisableLog('rdApp.*')
    rows = []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        ik = Chem.MolToInchiKey(m)
        rows.append(dict(Identifier=f'c{i}', MonoisotopicMass=Descriptors.ExactMolWt(m),
                         MolecularFormula=rdMolDescriptors.CalcMolFormula(m), SMILES=Chem.MolToSmiles(m),
                         InChI=Chem.MolToInchi(m), InChIKey1=ik[:14], InChIKey2=ik[15:25]))
    pd.DataFrame(rows).to_csv(path, index=False)


def _job(jar, d, sid, s):
    shift, mode = _SHIFT[s.adduct]
    mz = np.asarray(s.ms2_mzs, float); it = np.asarray(s.ms2_normalized_intensities, float)
    keep = mz < float(s.precursor_mz) + 1.0
    if keep.sum() == 0:
        return None
    pk = d / f'{sid}.txt'; out = d / f'{sid}.csv'; par = d / f'{sid}.par'
    np.savetxt(pk, np.c_[mz[keep], 999 * it[keep] / max(it[keep].max(), 1e-9)], fmt='%.5f')
    par.write_text('\n'.join([
        f'PeakListPath = {pk}', 'MetFragDatabaseType = LocalCSV', f'LocalDatabasePath = {d / "db.csv"}',
        f'NeutralPrecursorMass = {float(s.precursor_mz) - shift:.6f}', 'DatabaseSearchRelativeMassDeviation = 50',
        'FragmentPeakMatchAbsoluteMassDeviation = 0.003', 'FragmentPeakMatchRelativeMassDeviation = 10',
        f'PrecursorIonMode = {mode}', f'IsPositiveIonMode = {"True" if mode > 0 else "False"}',
        'MetFragScoreTypes = FragmenterScore', 'MetFragScoreWeights = 1.0', 'MetFragCandidateWriter = CSV',
        f'SampleName = {sid}', f'ResultsPath = {d}', 'MaximumTreeDepth = 2',
        'MetFragPreProcessingCandidateFilter = UnconnectedCompoundFilter', 'NumberThreads = 1']) + '\n')
    result = subprocess.run([MF_JAVA, '-Xmx1500m', '-jar', jar, str(par)], capture_output=True, timeout=300)
    if result.returncode:
        return None  # never mistake a stale CSV for a successful new call
    if not out.is_file():
        return None
    r = pd.read_csv(out)
    f = r.FragmenterScore.astype(float); top = f.max()
    return dict(zip(r.Identifier, (f / top).fillna(0) if top > 0 else f * 0))


def run_metfrag_stage():
    global MF_JAVA
    t0 = time.time()
    control = ROOT / 'submission.csv'
    if SUBMITTED != VARIANT or not control.is_file():
        MF_REPORT['status'] = f'skipped: control is {SUBMITTED}'; return False
    if (time.time() - NB_T0) / 3600 > MF_DEADLINE_H:
        MF_REPORT['status'] = 'skipped: session deadline'; return False
    jar = _metfrag_jar()
    if jar is None:
        MF_REPORT['status'] = 'skipped: MetFrag jar missing'; return False
    MF_JAVA, runtime = _runtime_preflight(jar)
    MF_REPORT['runtime'] = runtime
    if MF_JAVA is None:
        MF_REPORT['status'] = 'skipped: incompatible or unavailable runtime'
        return False
    shutil.copyfile(control, ROOT / 'submission_control_c472.csv')
    sub = pd.read_csv(control, dtype=str, keep_default_na=False)
    spec = pd.read_parquet(COMP / 'test.parquet')
    spec = spec[spec.adduct.isin(_SHIFT)].copy()
    spec['num_peaks'] = spec.ms2_mzs.map(len)   # the competition test file has no num_peaks column
    spec['tims'] = spec.instrument_type.astype(str).eq('timsTOF').astype(int)
    spec = spec.sort_values(['molecule_id', 'tims', 'num_peaks'], ascending=[True, False, False]).groupby('molecule_id').head(MF_SPECTRA)
    by_mol = {m: g for m, g in spec.groupby('molecule_id')}
    lists = {r.molecule_id: r.smiles.split(';') for r in sub.itertuples()}
    work, expected = [], {}
    for mid, smi in lists.items():
        if len(smi) < 3 or len(set(smi)) != len(smi) or mid not in by_mol:
            continue
        d = MF_ROOT / mid; d.mkdir(exist_ok=True); _db(smi, d / 'db.csv')
        jobs = [(mid, d, str(s.spectrum_id), s) for s in by_mol[mid].itertuples()]
        work += jobs; expected[mid] = len(jobs)
    budget = min(MF_BUDGET_S, (MF_STOP_H * 3600) - (time.time() - NB_T0))
    scores, finished, done, late = {}, collections.Counter(), 0, False
    with ThreadPoolExecutor(max(2, os.cpu_count() or 2)) as ex:
        futs = {ex.submit(_job, jar, d, sid, s): mid for mid, d, sid, s in work}   # submitted molecule by molecule
        for fu in as_completed(futs):
            mid = futs[fu]
            try:
                res = fu.result()
            except Exception:
                res = None
            finished[mid] += 1
            if res:
                scores.setdefault(mid, []).append(res); done += 1
            if time.time() - t0 > budget:
                late = True; break
        if late:
            for fu in futs: fu.cancel()
    complete = {m for m in scores if finished[m] == expected[m] and len(scores[m]) == expected[m]}
    if not complete:
        MF_REPORT.update(status='unapplied: no fully scored molecules', jobs=len(work),
                         done=done, molecules_scored=0, rows_reordered=0,
                         seconds=round(time.time() - t0))
        return False  # the control file remains byte-identical
    new_rows, moved = [], 0
    for mid, smi in lists.items():
        if mid not in complete:
            new_rows.append(';'.join(smi)); continue
        n = len(smi); tot = np.zeros(n); cnt = np.zeros(n)
        for res in scores[mid]:
            for k, v in res.items():
                tot[int(k[1:])] += v; cnt[int(k[1:])] += 1
        mf = np.divide(tot, cnt, out=np.zeros(n), where=cnt > 0)   # mean over the spectra that scored the candidate
        mrank = pd.Series(-mf).rank(method='average').values
        s = 1 / (3 + np.arange(1, n + 1)) + MF_WEIGHT / (3 + mrank)
        order = [0] + [i for i in np.argsort(-s, kind='stable') if i != 0]
        moved += int(order != list(range(n)))
        new_rows.append(';'.join(smi[i] for i in order))
    out = sub.copy(); out['smiles'] = new_rows
    # validation: same ids and order, same leader, same candidate set per row, 1..25 unique non-empty SMILES
    assert out.molecule_id.tolist() == SAMPLE_IDS
    for a, b in zip(out.smiles, sub.smiles):
        a, b = a.split(';'), b.split(';')
        assert a[0] == b[0] and sorted(a) == sorted(b) and 1 <= len(a) <= 25 and len(set(a)) == len(set(b)) and all(a)
    out.to_csv(ROOT / 'submission_metfrag.csv', index=False)
    shutil.copyfile(ROOT / 'submission_metfrag.csv', control)
    MF_REPORT.update(status=('applied (partial)' if late else 'applied') if moved else 'scored: no rank changes', jobs=len(work), done=done, molecules_scored=len(complete), rows_reordered=moved,
                     seconds=round(time.time() - t0), sha256=hashlib.sha256(control.read_bytes()).hexdigest())
    return True


try:
    MF_OK = run_metfrag_stage()
except Exception as exc:
    MF_OK = False; MF_REPORT['status'] = f'error: {type(exc).__name__}: {str(exc)[:300]}'
    if (ROOT / 'submission_control_c472.csv').is_file():
        shutil.copyfile(ROOT / 'submission_control_c472.csv', ROOT / 'submission.csv')
(ROOT / 'metfrag_report.json').write_text(json.dumps(MF_REPORT, indent=1))
print('METFRAG STAGE', json.dumps(MF_REPORT))
