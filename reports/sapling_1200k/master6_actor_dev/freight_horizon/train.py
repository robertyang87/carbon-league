"""Fit a fixed-budget, deployable public freight detector and its distance/cargo control."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from predictor import NAMES,ensemble
from data import HERE,CACHE,sha

def save(f,obj):
    tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=1,allow_nan=False)+'\n');os.replace(tmp,f)

def serialize(m,features):
    forest=[]
    for predictor in m._predictors:
        nodes=[]
        for n in predictor[0].nodes:
            nodes.append([-1 if n['is_leaf'] else int(features[int(n['feature_idx'])]),float(n['num_threshold']),int(n['left']),int(n['right']),float(n['value'])]);
        forest.append(nodes)
    return {'offset':float(m._baseline_prediction.reshape(-1)[0]),'trees':forest}

def fit(x,y,w,columns):
    Z=x[:,columns];args=dict(max_iter=70,max_depth=3,max_leaf_nodes=8,learning_rate=.065,l2_regularization=2.,early_stopping=False,random_state=910739)
    a=HistGradientBoostingRegressor(loss='squared_error',**args).fit(Z,y[:,0],sample_weight=w)
    b=HistGradientBoostingClassifier(loss='log_loss',**args).fit(Z,y[:,1].astype(int),sample_weight=w)
    return {'features':NAMES,'principal':serialize(a,columns),'any':serialize(b,columns)},(a,b,columns)

def apply(bundle,x):return np.clip([ensemble(bundle['principal'],row) for row in x],0,1),1/(1+np.exp(-np.array([ensemble(bundle['any'],row) for row in x])))

def by_map_mean(values):return {k:float(np.mean([x[k] for x in values])) for k in values[0]} if values else {}

def amount(arr):return round(float(arr),6)

def metrics(x,y,uid,mask,forecast):
    f,p=[z[mask] for z in forecast];xx=x[mask];truth=y[mask];ids=uid[mask]
    bymap=[];groups=np.unique(ids)
    for g in groups:
        take=ids==g;v=truth[take];frac=f[take];p1=p[take];cash=xx[take,0]*frac
        bymap.append(dict(principal_mae=float(np.abs(cash-v[:,3]).mean()),principal_signed_bias=float((cash-v[:,3]).mean()),
                          fraction_mse=float(np.mean((v[:,0]-frac)**2)),brier=float(np.mean((v[:,1]-p1)**2)),
                          logloss=float(np.mean(-(v[:,1]*np.log(np.clip(p1,1e-8,1))+(1-v[:,1])*np.log(np.clip(1-p1,1e-8,1))))),
                          actual_frac=float(v[:,0].mean()),expected_frac=float(frac.mean()),actual_bank=float(v[:,1].mean()),
                          actual_gross=float(v[:,2].mean()),actual_principal=float(v[:,3].mean()),expected_principal=float(cash.mean())))
    return dict(rows=int(mask.sum()),maps=len(groups),deposits=int(truth[:,1].sum()),map_weighted=bymap and by_map_mean(bymap),
                by_map={str(int(g)):z for g,z in zip(groups,bymap)})

def main():
    cards=json.loads((HERE/'dataset.json').read_bytes())['rows'];X=[];Y=[];G=[];datafiles={};op={};group={};games_with_no_rows=[]
    for r in cards:
        path=CACHE/(r['id']+'.npz');buf=path.read_bytes();digest=sha(buf)
        if digest!=r['data_sha256']:raise ValueError('shard changed')
        with np.load(path) as d:
            if not len(d['x']):games_with_no_rows.append(r['id']);continue
            X.append(d['x']);Y.append(d['y']);G.append(np.full(len(d['x']),int(r['gid'])));group[r['gid']]=r['split'];op[r['gid']]=r['opponent'];datafiles[r['id']]=digest
    x,y,g=map(np.concatenate,(X,Y,G));train=np.array([group[i]=='train' for i in g]);gtrain=g[train];unique,counts=np.unique(gtrain,return_counts=True)
    count=dict(zip(unique,counts));w=np.array([len(gtrain)/len(unique)/count[i] for i in gtrain]);assert np.isclose(w.sum(),len(w))
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('nonfinite data')
    receipt=dict(sha256_protocol=sha((HERE/'protocol.json').read_bytes()),data=sha((HERE/'dataset.json').read_bytes()),train_rows=len(w),train_maps=len(unique),sklearn=sklearn.__version__,features=NAMES)
    models={};estimators={}
    for key,cols in [('rich',list(range(x.shape[1]))),('distance_and_cargo',[0,1])]:
        obj,sk=fit(x[train],y[train],w,cols);obj.update(receipt=receipt,variant=key);models[key]=obj;estimators[key]=sk
        save(HERE/('model.json' if key=='rich' else 'baseline.json'),obj)
    print('MODEL_FROZEN',sha((HERE/'model.json').read_bytes()),flush=True)
    known={op[i] for i in unique};splits={s:np.array([group[i]==s for i in g]) for s in ('train','val','holdout')};splits['holdout_unseen_opponent']=splits['holdout'] & np.array([op[i] not in known for i in g]);splits['holdout_seen_opponent']=splits['holdout'] & ~splits['holdout_unseen_opponent'];forecasts={};parity={}
    for key,bundle in models.items():
        a,b,cols=estimators[key];f=np.clip(a.predict(x[:,cols]),0,1);p=b.predict_proba(x[:,cols])[:,1];forecasts[key]=(f,p)
        # Verify the lightweight exported forest against sklearn for every held-out example.
        rows=np.flatnonzero(splits['holdout']|splits['val']);ff,pp=apply(bundle,x[rows]);error=max(float(abs(ff-f[rows]).max()),float(abs(pp-p[rows]).max()));parity[key]=error
        if error>1e-10:raise ValueError('standalone predictor differs '+str(error))
    res={}
    for s,mask in splits.items():
        res[s]={'names':sorted({op[i] for i in np.unique(g[mask])}), 'scores':{k:metrics(x,y,g,mask,v) for k,v in forecasts.items()}}
        for field in ('principal_mae','brier','fraction_mse'):
            richer=res[s]['scores']['rich']['by_map'];base=res[s]['scores']['distance_and_cargo']['by_map'];diff=np.array([richer[i][field]-base[i][field] for i in sorted(richer)])
            if not len(diff): continue
            rng=np.random.default_rng(910741);bounds=np.quantile(diff[rng.integers(0,len(diff),size=(10000,len(diff)))].mean(1),[.025,.975]);res[s].setdefault('paired_map_difference_rich_minus_base',{})[field]=dict(mean=float(diff.mean()),bootstrap_ci=bounds.tolist(),better=int((diff<0).sum()),maps=len(diff))
    save(HERE/'results.json',dict(receipt=receipt,unrepresented_maps=games_with_no_rows,prediction_reader_max_abs_error=parity,results=res))
    print('held',res['holdout']['paired_map_difference_rich_minus_base'],flush=True)

if __name__=='__main__':main()
