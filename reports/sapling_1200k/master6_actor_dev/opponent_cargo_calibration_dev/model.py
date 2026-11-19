"""Enemy loaded collectors from the observer's last eleven public views, in physical action coordinates."""
from pathlib import Path
import importlib.util
import json
import numpy as np

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent;BASE=ACTOR.parent;RULES=BASE.parent/'master_family/tactical_rules'


def _module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


intent=_module('sapling_public_intent',RULES/'opponent_intent.py')
freight=_module('sapling_public_courier',ACTOR/'freight_horizon/predictor.py')
legacy=json.loads((RULES/'opponent_intent.json').read_text())
LEGACY=np.array(legacy['weights'])
TARGETS=('bank3','bank5','contest3','contest5')
EXTRA=('has_t10','home_progress_t10','cargo_delta_t10','same_cell_t10','lighter_enemy_nearest','heavier_enemy_nearest',
       'loaded_enemy_distance','log_enemy_cash','log_own_cash','n_carriers','previous_home_progress','p_repeat','p_homeward','p_toward_victim','prior_entropy')
FEATURES=tuple(freight.NAMES)+EXTRA


def public_rows(history):
    if not history or len(history)>11:raise ValueError('supply 1..11 ordered public observations')
    obs=history[-1];s=obs['player'];t=obs['step']
    if any(o['player']!=s or o['step']!=t-len(history)+1+i for i,o in enumerate(history)):
        raise ValueError('history has a gap or mixes seats')
    old=history[-2] if len(history)>1 else None
    ids,roles,z=intent.features(obs,old)
    if not len(ids):return []
    prior=intent.probabilities(z,roles,LEGACY)
    retained=freight.public_features(obs,history[:-1])
    focal=obs['players'][s];op=obs['players'][1-s];of=op[2];bases=list(op[1].values());H=freight.DIST
    recent={v['step']:v['players'][1-s][2] for v in history}
    def nearest(p,cells):return min((H[p][c] for c in cells),default=15)
    records=[]
    for i,uid in enumerate(ids):
        if uid not in retained:continue
        pos,cargo,k=of[uid]; assert cargo>0 and k=='COLLECTOR'
        home=nearest(pos,bases);late=recent.get(t-10,{}).get(uid);prev=recent.get(t-1,{}).get(uid);three=recent.get(t-3,{}).get(uid)
        prev_move=int(intent.move_between(prev[0],pos)) if prev is not None else 5
        if prev_move>4:prev_move=-1
        light=[p for p,c,k in focal[2].values() if k=='COLLECTOR' and c<=cargo]
        heavy=[p for p,c,k in focal[2].values() if k=='COLLECTOR' and c>cargo]
        loaded=[p for p,c,k in focal[2].values() if k=='COLLECTOR' and c>0]
        dst=intent.DEST[pos];pr=prior[i]
        px=(*retained[uid],float(late is not None),0 if not late else nearest(late[0],bases)-home,
            0 if not late else cargo-late[1],0 if not late else int(late[0]==pos),nearest(pos,light),nearest(pos,heavy),
            nearest(pos,loaded),np.log1p(op[0]),np.log1p(focal[0]),len(retained),0 if not prev else nearest(prev[0],bases)-home,
            0 if prev_move<0 else float(pr[prev_move]),sum(pr[a] for a,d in enumerate(dst) if nearest(d,bases)<home),
            sum(pr[a] for a,d in enumerate(dst) if int(d) in heavy),-float(np.sum(pr*np.log(pr+1e-15))))
        group=int(three is not None and nearest(three[0],bases)>home)
        records.append((uid,np.asarray(px,dtype='f'),pr.copy(),prev_move,group))
    return records


def move_prob(raw,group,temp):
    log=np.log(np.maximum(raw,1e-12))/temp[np.asarray(group,dtype=int),None];e=np.exp(log-log.max(axis=1,keepdims=True));return e/e.sum(1,keepdims=True)


def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-60,60)))


class Forecast:
    def __init__(self,head_path=None):
        with np.load(head_path or HERE/'head.npz',allow_pickle=False) as z:self.w={k:z[k] for k in z.files}

    def arrays(self,x,prior,group):
        w=self.w;q=move_prob(prior,group,w['temperature']);scaled=np.clip((x-w['mu'])/w['scale'],-10,10)
        raw=scaled@w['weights']+w['bias'];p=sigmoid(raw*w['platt_slope']+w['platt_bias'])
        p[:,1]=np.maximum(p[:,0],p[:,1]);p[:,3]=np.maximum(p[:,2],p[:,3])
        return q,p

    def predict(self,history):
        rows=public_rows(history)
        if not rows:return {}
        q,p=self.arrays(np.stack([r[1] for r in rows]),np.stack([r[2] for r in rows]),[r[4] for r in rows])
        return {r[0]:{'move':q[i].tolist(),**dict(zip(TARGETS,map(float,p[i])))} for i,r in enumerate(rows)}
