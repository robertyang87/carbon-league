from pathlib import Path
from collections import defaultdict
import sys
import numpy as np
HERE=Path(__file__).resolve().parent;ACTOR=HERE.parent
sys.path.insert(0,str(ACTOR/'tactical_exchange_dev'))
from measure import sr
from decision import update_memory,values,predicted,FEATURES
from turning_point_ledger import CashEvents

def extract(g):
    raw=sr.load(sr.ROOT/g['path']);cfg=raw['game_configuration'];sim=sr.FastCarbon(cfg).reset(seed=raw['seed']);sim.cash=CashEvents(sim.cash)
    mem=[{},{}];active={};rows=[]
    for t,cmd in enumerate(raw['actions']):
        obs=[sim.obs(s) for s in (0,1)]
        for s in (0,1):update_memory(obs[s],mem[s])
        sim.cash.events.clear();sim.step(*cmd)
        for kind,s,val,*a in sim.cash.events:
            if kind=='plant':
                uid,cell=a;f=values(obs[s],mem[s],uid,cell,cfg)
                if f is None:continue
                tree=next(v for v in sim.trees.values() if v.pos==cell)
                r=dict(t=t,seat=s,tree=tree.id,cell=cell,uid=uid,x=list(map(float,f)),fee=-val,ground=0.,tax=0.,end=None)
                rows.append(r);active[tree.id]=r
            elif kind=='tree_absorption' and a[0] in active:
                r=active[a[0]]
                if r['seat']==s:r['ground']+=val-a[2];r['tax']+=a[2]
        for k,r in list(active.items()):
            if k not in sim.trees or sim.trees[k].player!=r['seat']:
                r['end']=sim.step_count;del active[k]
    assert list(sim.cash)==raw['official_score_by_seat']
    return dict(seed=g['seed'],source=g['path'],seat=g['seat'],rows=rows)

def run():
    games=[extract(g) for g in sr.load(ACTOR/'tactical_exchange_dev/paired.json.gz')]
    sr.retain(HERE/'purchases.json.gz',games)
    train=[r for g in games if g['seed']<=621758326 for r in g['rows']]
    x=np.array([[1,*r['x']] for r in train]);y=np.array([r['ground'] for r in train])
    penalty=8*np.eye(x.shape[1]);penalty[0,0]=0.
    w=np.linalg.solve(x.T@x+penalty,x.T@y)
    sr.retain(HERE/'weights.json',dict(features=FEATURES,coef=w.tolist(),train_count=len(y),train_seeds=sorted({g['seed'] for g in games if g['seed']<=621758326})))
    metrics=[]
    for seed in (621758327,621758328):
        rows=[r for g in games if g['seed']==seed for r in g['rows']]
        a=np.array([r['ground'] for r in rows]);p=np.array([predicted(r['x'],w) for r in rows]);q=np.array([r['fee'] for r in rows])
        metrics.append(dict(seed=seed,n=len(rows),mae=float(np.mean(abs(p-a))),constant_mae=float(np.mean(abs(y.mean()-a))),ground=float(np.mean(a)),tax=float(np.mean([r['tax'] for r in rows])),corr=float(np.corrcoef(a,p)[0,1]),payback_accuracy=float(np.mean((a>q)==(p>q)))))
    sr.retain(HERE/'fit.json',metrics);print(len(y),metrics)
if __name__=='__main__':run()
