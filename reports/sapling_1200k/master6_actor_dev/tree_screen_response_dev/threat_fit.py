"""Estimate impending loss from the pre-approach state, including feints that never enter a tree."""
from collections import defaultdict
import numpy as np
from screen_trial import HERE,sr
from pathlib import Path
import sys
sys.path.insert(0,str(HERE.parent/'intercept_response_dev'))
from forecast import _D, _TO, _DIRECTION

NAMES=['tree age','turn','tree remaining turns','last tree payment','enemy cash','own collectors within two',
       'enemy collectors within two','nearest friendly collector','enemy planters within two','enemy planter arrived from outside']

def rows(game, planned):
    raw=sr.load(sr.ROOT/game['path']);s=game['seat'];sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
    recent={}; pending=defaultdict(list); output=[]
    for t,pair in enumerate(raw['actions']):
        obs=sim.obs(s);own,enemy=obs['players'][s],obs['players'][1-s]
        for tree,(cell,age) in own[3].items():
            if age>=sim.config['treeLifespan']-2 or enemy[0]<sim.config['seizeCost'] or t+2>299:continue
            oc=[_D[cell,v[0]] for v in own[2].values() if v[2]=='COLLECTOR']
            ec=[_D[cell,v[0]] for v in enemy[2].values() if v[2]=='COLLECTOR']
            ep=[_D[cell,v[0]] for v in enemy[2].values() if v[2]=='PLANTER']
            for uid,(p,_,kind) in enemy[2].items():
                if kind!='PLANTER' or _D[p,cell]!=1:continue
                pay=(obs.get('trees',{}).get(tree) or [None,0])[1] or 0
                x=[age/50,t/299,min(299-t,50-age)/50,pay/5,min(enemy[0]/100,4),sum(v<=2 for v in oc),
                   sum(v<=2 for v in ec),min(min(oc,default=15),8)/4,sum(v<=2 for v in ep),
                   uid in recent and _D[recent[uid][0],cell]>1]
                if planned:
                    lands=[int(_D[int(_TO[v[0],_DIRECTION.index(pair[s].get(k))]),cell])
                           for k,v in own[2].items() if v[2]=='COLLECTOR']
                    x.extend([min(min(lands,default=15),8)/4,sum(d<=1 for d in lands),sum(d<=2 for d in lands)])
                pending[t+1].append(dict(t=t,tree=tree,foe=uid,x=[float(v) for v in x]))
        recent=enemy[2];sim.step(*pair)
        for r in pending.pop(t,[]):
            tree=sim.trees.get(r['tree']);r['seized']=bool(tree and tree.player==1-s);output.append(r)
    assert list(sim.cash)==raw['official_score_by_seat']
    return output

def run(planned):
    tag="planned_" if planned else ""
    games=sr.load(HERE.parent/'tactical_exchange_dev/paired.json.gz')
    allrows=[dict(**r,seed=g['seed'],seat=g['seat'],source=g['path']) for g in games for r in rows(g,planned)]
    train=[r for r in allrows if r['seed']<=621758326]
    test=[r for r in allrows if r['seed']>621758326]
    x=np.array([[1]+r['x'] for r in train]);y=np.array([r['seized'] for r in train]);w=np.zeros(x.shape[1])
    penalty=np.diag([0]+[2]*(x.shape[1]-1))
    for _ in range(30):
        prob=1/(1+np.exp(-np.clip(x@w,-35,35)))
        diff=np.linalg.solve(x.T@((prob*(1-prob))[:,None]*x)+penalty,x.T@(y-prob)-penalty@w)
        w+=diff
        if np.max(np.abs(diff))<1.e-8:break
    base=float(y.mean());scores=[]
    for seed in sorted({r['seed'] for r in test}):
        rs=[r for r in test if r['seed']==seed];xt=np.array([[1]+r['x'] for r in rs]);actual=np.array([r['seized'] for r in rs])
        p=1/(1+np.exp(-xt@w))
        idx=np.argsort(-p,kind='stable')[:max(1,len(p)//4)]
        scores.append(dict(seed=seed,opportunities=len(rs),losses=int(actual.sum()),brier=float(((p-actual)**2).mean()),
                          constant_brier=float(((base-actual)**2).mean()),top_quarter=len(idx),top_losses=int(actual[idx].sum())))
        for r,v in zip(rs,p):r['forecast']=float(v)
    sr.retain(HERE/(tag+'threat_model.json'),dict(names=['intercept']+NAMES+(['planned nearest defender','planned defenders within one','planned defenders within two'] if planned else []),w=w.tolist(),train_steps=len(train),train_rate=base,holdout=scores))
    sr.retain(HERE/(tag+'threat_observations.json.gz'),allrows)
    print(scores)
if __name__=='__main__':run(sys.argv[1:]==['planned'])
