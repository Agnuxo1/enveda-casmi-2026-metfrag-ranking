"""Apply the experimental fixed blend to its exact numeric feature contract.

Use only model files produced by experimental_weighting.py; joblib files are
executable Python serialization. This is not a competition submission generator.
"""
import argparse
from pathlib import Path
import numpy as np


def predict(features, model_dir):
    import joblib
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 31 or not np.isfinite(x).all():
        raise ValueError('Expected finite [candidate,31] features in original sim-ranker order')
    scores = []
    for name in ['M4_query_balanced', 'M5_positive5']:
        model = joblib.load(Path(model_dir)/(name+'.joblib'))
        scores.append(model.predict_proba(x)[:,1])
    return .5*scores[0]+.5*scores[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features',type=Path,required=True,help='Numpy .npy array,31 ordered features')
    ap.add_argument('--models',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    np.save(a.out,predict(np.load(a.features,allow_pickle=False),a.models))


if __name__=='__main__':main()
