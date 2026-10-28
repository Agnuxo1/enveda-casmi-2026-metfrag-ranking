"""New ranker weighting interventions on real-spectrum features in controlled retrieval simulations."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,csv,json,hashlib,time,datetime
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import sklearn

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output'));ap.add_argument('--export-models',action='store_true');args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True)
 z=np.load(args.data,allow_pickle=False);X=z['X'][:,:31].astype(np.float32);Y=z['Y'].astype(int);G=z['G'];Q=z['Q'];S=z['S']
 train=Q!=0;test=Q==0
 assert set(np.unique(G[train])).isdisjoint(np.unique(G[test]))
 assert set(np.unique(Y))<={0,1} and np.isfinite(X).all()
 groups,counts=np.unique(G[train],return_counts=True);cnt=dict(zip(groups,counts));balanced=np.array([1/cnt[g] for g in G[train]])
 positive=np.where(Y[train]==1,5.,1.);weights={'baseline':np.ones(train.sum()),'M4_query_balanced':balanced/balanced.mean(),'M5_positive5':positive/positive.mean()}
 scores={};seconds={};exports={}
 for name,w in weights.items():
  t=time.monotonic();model=HistGradientBoostingClassifier(max_depth=6,max_iter=150,learning_rate=.05,min_samples_leaf=80,l2_regularization=1.,random_state=20261028,early_stopping=False)
  model.fit(X[train],Y[train],sample_weight=w);scores[name]=model.predict_proba(X[test])[:,1];seconds[name]=time.monotonic()-t
  if args.export_models:
   import joblib
   path=args.out/(name+'.joblib');joblib.dump(model,path)
   restored=joblib.load(path).predict_proba(X[test])[:,1]
   assert np.array_equal(restored,scores[name]),'Export changed predictions'
   exports[name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'prediction_roundtrip_exact':True}
  print(json.dumps({'model':name,'seconds':seconds[name]}),flush=True)
 scores['M6_half_blend']=.5*scores['M4_query_balanced']+.5*scores['M5_positive5']
 y=Y[test];g=G[test];s=S[test];rows=[]
 for group in np.unique(g):
  ix=np.flatnonzero(g==group);assert len(np.unique(s[ix]))==1
  for name,p in scores.items():
   order=ix[np.argsort(-p[ix],kind='stable')];hit=np.flatnonzero(y[order]>0);rank=int(hit[0]+1) if len(hit) else None
   rows.append(dict(group=int(group),molecule=int(group)//2,regime=int(s[ix[0]]),method=name,candidates=len(ix),rank=rank,rr25=1/rank if rank is not None and rank<=25 else 0.,top1=int(rank==1),recall25=int(rank is not None and rank<=25)))
 with (args.out/'casmi_group_metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 import pandas as pd
 d=pd.DataFrame(rows);summary=d.groupby(['regime','method'])[['rr25','top1','recall25']].mean().reset_index().to_dict(orient='records');contrasts=[];rng=np.random.default_rng(20261028)
 for regime in [0,1,'all']:
  ds=d if regime=='all' else d[d.regime==regime]
  wide=ds.pivot(index=['molecule','group'],columns='method',values='rr25')
  molecules=wide.index.get_level_values('molecule').unique();ids=np.array(wide.index.get_level_values('molecule'));ix={m:np.flatnonzero(ids==m) for m in molecules}
  boots=[np.concatenate([ix[m] for m in rng.choice(molecules,len(molecules),replace=True)]) for _ in range(2000)]
  for name in list(scores)[1:]:
   delta=(wide[name]-wide.baseline).to_numpy()
   contrasts.append({'regime':regime,'method':name,'n_groups':len(wide),'n_molecules':len(molecules),'delta_mrr25':float(delta.mean()),'ci95_molecule':[float(x) for x in np.quantile([delta[ii].mean() for ii in boots],[.025,.975])],'positive_groups':int((delta>0).sum()),'negative_groups':int((delta<0).sum())})
 out={'kind':'new_CPU_weighting_and_blending_interventions_on_retrieval_simulation','data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),'train_rows':int(train.sum()),'test_rows':int(test.sum()),'train_groups':len(np.unique(G[train])),'represented_test_groups':len(np.unique(g)),'expected_Q0_groups_from_dataset_description':500,'train_test_group_overlap':0,'features':31,'summary':summary,'contrasts':contrasts,'fit_seconds':seconds,'sklearn':sklearn.__version__,'limits':['Derived features of observed public spectra, with simulated library-removal regimes; not an official CASMI score.','Q0 Enveda examples reserved; Q1-6 training.','Group IDs disjoint; exact chemical identity overlap across public libraries cannot be independently ruled out without raw structure metadata.','Only represented queries can be scored; missing groups are disclosed.','No test selection, no Kaggle submission.'],'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 out['model_exports']=exports
 out['feature_contract']='First 31 columns of the specified sim_rank_rows_nofp.npz X array, in original order; not interchangeable with MetFrag or notebook features.'
 (args.out/'casmi_weighting_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({'summary':summary,'contrasts':contrasts,'model_exports':exports}),flush=True)
if __name__=='__main__':main()
