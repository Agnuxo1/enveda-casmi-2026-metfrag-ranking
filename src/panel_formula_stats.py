import os
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
d=pd.read_parquet(os.environ.get('CASMI_PANEL_DIR', 'data/catalogue') + r'/catalogue_experiment/labelled_candidate_scores.parquet')
def f(s):
    m=Chem.MolFromSmiles(s); return CalcMolFormula(m) if m else None
d['formula']=d.smiles.map(f)
rows=[]
for m,g in d.groupby('molecule_id'):
    g=g.sort_values('baseline_rank'); tf=g.loc[g.label==1,'formula']
    rows.append(dict(nform=g.formula.nunique(), top_form_true=(len(tf)>0 and tf.iloc[0]==g.formula.iloc[0]), has_true=len(tf)>0,
                     frac_same_as_top=(g.formula==g.formula.iloc[0]).mean()))
r=pd.DataFrame(rows); print(r.describe()); print('true formula == rank1 formula:', r.top_form_true.mean())
