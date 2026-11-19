"""Fit observed next-turn settlement and the realized net earnings on the first holding period."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from features import PlanterRouteState, _ROUTE_DEST, _ROUTE_FIELDS

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
CACHE=Path('/workspace/sapling-all-top5-20261119')
MOVES={'UP':1,'RIGHT':2,'DOWN':3,'LEFT':4}
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def write(name,x): (HERE/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

def select():
    manifest=json.loads((BASE/'all_top5/manifest.json').read_bytes())
    exclude={g['seed'] for g in json.loads((BASE/'latest5_inventory.json').read_bytes())['games']}|{621456920}
    groups=[[],[]]; seeds=set()
    for g in sorted(manifest['games'],key=lambda g:sha('route-map-1|'+str(g['seed'])+g['id'])):
        if g['seed'] in exclude or g['seed'] in seeds:continue
        seeds.add(g['seed'])
        ext=[name for name,ident in zip(g['seat_names'],g['seat_strategy_ids']) if ident is None]
        fresh=any(int(sha('route-opponent-1|'+name),16)%5==0 for name in ext)
        groups[int(fresh)].append({k:g[k] for k in ['id','seed','seat_names','seat_strategy_ids','sha256','path']})
    regular,novel=groups
    split={'train':regular[:40],'validation':regular[40:50],'test_seen_pool':regular[50:56],'test_unseen_opponent':novel[:6]}
    allg=[g for gs in split.values() for g in gs]
    if len(set(g['seed'] for g in allg))!=62:raise ValueError('map overlap')
    write('split.json',split);return split

def rows(g):
    fpath=CACHE/g['path'];b=fpath.read_bytes()
    if hashlib.sha256(b).hexdigest()!=g['sha256']:raise ValueError('source changed')
    tape=json.loads(gzip.decompress(b))
    with gzip.open(CACHE/'plant_work'/(g['id']+'.gz')) as f:credit=json.load(f)
    if credit['id']!=g['id'] or credit['replay_sha256']!=g['sha256']:raise ValueError('ledger mismatch')
    receipts={(a[0],a[1],a[2],a[4]):a for a in credit['purchases']}
    sts=[PlanterRouteState(),PlanterRouteState()];choices=[];frames=tape['steps']
    for t in range(min(196,len(frames)-2)):
        board=frames[t][0]['observation']
        if len(board.get('players',[]))!=2:raise ValueError('missing shared fields')
        for seat,st in enumerate(sts):
            st.observe(board,seat,frames[t][seat]['observation']['carbon'],tape['configuration'],t)
            if t<45:continue
            future=frames[t+1][seat]['action'] or {}
            for uid,(pos,_,typ) in board['players'][seat][2].items():
                if typ!='PLANTER' or future.get(uid) not in MOVES:continue
                dst=int(_ROUTE_DEST[pos,MOVES[future[uid]]])
                feat=st.features(dst)
                if feat is None:continue
                rec=receipts.get((t+1,seat,uid,dst));net=rec[11]+rec[13]-rec[6] if rec else 0.
                choices.append([feat[0],int(bool(rec)),float(net)])
    if len(choices)>300:
        idx=np.random.default_rng(int(sha('route-cap-1|'+str(g['seed']))[:16],16)).choice(len(choices),300,replace=False)
        choices=[choices[j] for j in sorted(idx)]
    return choices

def pack(games):
    x=[];y=[];v=[];w=[];maps=[]
    for g in games:
        obs=rows(g)
        if not obs: continue
        for a,b,c in obs:x.append(a);y.append(b);v.append(c);w.append(1./len(obs));maps.append(g['id'])
    w=np.array(w);return np.array(x),np.array(y),np.array(v),w/w.sum(),maps

def fit(dataset):
    x,y,v,w,_=dataset;mean=w@x;std=np.maximum(np.sqrt(w@((x-mean)**2)),.05)
    z=(x-mean)/std;D=np.c_[z,np.ones(len(z))]
    reg=np.r_[np.ones(z.shape[1]),0.]
    def loss(b):
        l=D@b;p=expit(l)
        return float(w@(np.logaddexp(0,l)-y*l)+.5*.03*(reg@b**2)),D.T@(w*(p-y))+.03*reg*b
    prob=minimize(loss,np.zeros(D.shape[1]),method='L-BFGS-B',jac=True)
    if not prob.success:raise ValueError(prob.message)
    mask=y==1;wp=w[mask]/sum(w[mask]);A=D[mask];target=v[mask]
    cash=np.linalg.solve(A.T@(A*wp[:,None])+.15*np.diag(reg),A.T@(wp*target))
    return {'fields':_ROUTE_FIELDS,'mean':mean.tolist(),'scale':std.tolist(),'purchase_weights':prob.x[:-1].tolist(),'purchase_bias':float(prob.x[-1]),'net_weights':cash[:-1].tolist(),'net_bias':float(cash[-1]),'training_purchase_rate':float(w@y),'training_mean_net':float(w@v)}

def result(ds,m):
    x,y,v,w,maps=ds
    z=(x-m['mean'])/m['scale'];p=expit(z@m['purchase_weights']+m['purchase_bias']);h=p*np.clip(z@m['net_weights']+m['net_bias'],-150,250)
    base=m['training_purchase_rate'];v0=m['training_mean_net']
    def ll(p):return float(-w@(y*np.log(p)+(1-y)*np.log1p(-p)))
    def grp(mask):return {'n':int(sum(mask)),'paid_rate':float(w[mask]@y[mask]/w[mask].sum()),'realized_net':float(w[mask]@v[mask]/w[mask].sum()),'predicted_net':float(w[mask]@h[mask]/w[mask].sum())}
    return {'games':len(set(maps)),'observations':len(y),'purchases':int(y.sum()),'brier':float(w@(p-y)**2),'brier_constant':float(w@(base-y)**2),'logloss':ll(p),'logloss_constant':ll(base),'net_mae':float(w@abs(h-v)),'net_mae_constant':float(w@abs(v-v0)),'net_rmse':float(np.sqrt(w@(h-v)**2)),'net_rmse_constant':float(np.sqrt(w@(v-v0)**2)),'top_forecast_quartile':grp(h>=np.quantile(h,.75)),'bottom_forecast_quartile':grp(h<=np.quantile(h,.25))}
if __name__=='__main__':
    split=select();datasets={}
    for s,gs in split.items():
        datasets[s]=pack(gs);print(s,len(datasets[s][0]),flush=True)
        if s=='train':
            model=fit(datasets[s]);write('weights.json',model)
    write('holdout.json',{s:result(v,model) for s,v in datasets.items()})
