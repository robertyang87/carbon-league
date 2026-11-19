"""Fit conditional realized duration/receipts, calibrate on independent maps and export NumPy weights."""
import argparse
from pathlib import Path
import platform
import hashlib
import json
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from dataset import HERE, FEATURES, SPLITS, read
from model import Predictor


def export(models, calibration):
    w = {k:[] for k in ['value','left','right','feature','threshold']}
    offsets = []
    for m in models:
        boundaries = []
        for row in m._predictors:
            tree = row[0].nodes; offset=len(w['value'])
            for node in tree:
                w['value'].append(node['value'])
                w['left'].append(-1 if node['is_leaf'] else offset+node['left'])
                w['right'].append(-1 if node['is_leaf'] else offset+node['right'])
                w['feature'].append(node['feature_idx'])
                w['threshold'].append(node['num_threshold'])
            boundaries.append([offset,len(w['value'])])
        offsets.append(boundaries)
    return {**{k:np.asarray(v) for k,v in w.items()},'offset':np.array(offsets),
            'base':np.asarray([m._baseline_prediction.item() for m in models]),'calib':calibration}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(cache, predictions):
    a=np.load(cache); x=a['x'].astype(np.float64); y=a['y'][:,:2]
    ids=a['map']; maps=read(HERE/'maps.json.gz')['maps']
    split=np.array([SPLITS.index(r['split']) for r in maps])[ids]
    train=split==0; cal=split==1; kind=x[:,0].astype(int)
    models=[]; calibrations=np.zeros((2,2,2))
    means=np.zeros_like(y); ridge=np.zeros_like(y); raw=np.zeros_like(y); predictions_all=np.zeros_like(y)
    for q in range(2):
        m=HistGradientBoostingRegressor(max_iter=75,max_leaf_nodes=15,learning_rate=.07,min_samples_leaf=60,l2_regularization=5,
                                       early_stopping=False,random_state=19)
        m.fit(x[train],y[train,q]); models.append(m)
        raw[:,q]=m.predict(x)
        for k in (0,1):
            kt=train & (kind==k); kc=cal & (kind==k); kk=kind==k
            mu=raw[kc,q].mean(); target=y[kc,q].mean()
            slope=np.clip(np.sum((raw[kc,q]-mu)*(y[kc,q]-target))/np.sum((raw[kc,q]-mu)**2),0,2)
            offset=target-slope*mu; calibrations[q,k]=[offset,slope]
            means[kk,q]=y[kt,q].mean()
            xm=x[kt].mean(0); sd=np.maximum(.05,x[kt].std(0)); z=(x[kt]-xm)/sd
            b=np.linalg.solve(z.T@z+30*np.eye(z.shape[1]),z.T@(y[kt,q]-y[kt,q].mean()))
            r=y[kt,q].mean()+((x[kk]-xm)/sd)@b
            ridge[kk,q]=np.maximum(0,r)
            predictions_all[kk,q]=np.maximum(0.,offset+slope*raw[kk,q])
    for mat in (means,ridge,predictions_all):
        mat[:,0]=np.minimum(mat[:,0],x[:,FEATURES.index('remaining_turns')])
    weights=export(models,calibrations)
    np.savez_compressed(HERE/'weights.npz',**weights)
    rng=np.random.default_rng(54021); subset=rng.choice(len(x),size=min(1024,len(x)),replace=False)
    portable=Predictor(weights).predict_array(x[subset])
    parity=float(np.abs(portable-predictions_all[subset]).max()); assert parity<1e-8,parity
    np.savez_compressed(predictions, pred=predictions_all, raw=raw, mean=means, ridge=ridge)
    md=dict(sklearn=sklearn.__version__,python=platform.python_version(),numpy=np.__version__,weight_sha256=sha(HERE/'weights.npz'),
            protocol_sha256=sha(HERE/'protocol.json'),maps_sha256=sha(HERE/'maps.json.gz'),portable_max_absolute_error=parity,
            parity_rows=len(subset),calibration=calibrations.tolist(),fit_events=int(train.sum()),calibration_events=int(cal.sum()))
    (HERE/'fit.json').write_text(json.dumps(md,indent=2)+'\n')
    print(md,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cache',type=Path);p.add_argument('predictions',type=Path);a=p.parse_args()
    fit(a.cache,a.predictions)
