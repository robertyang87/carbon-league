from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ACTOR=HERE.parent
sys.path[:0]=[str(ACTOR/'tactical_exchange_dev'),str(ACTOR/'tree_screen_response_dev')]
from measure import sr
from payoff import price,freight,tree_features,HORIZON
from guard_tree import _D
from turning_point_ledger import CashEvents
from collections import defaultdict
import numpy as np

def collect(g):
    raw=sr.load(sr.ROOT/g['path']);s=g['seat'];cfg=raw['game_configuration']
    sim=sr.FastCarbon(cfg).reset(seed=g['seed']);sim.cash=CashEvents(sim.cash)
    jobs=[[],[]];flows=[defaultdict(list),defaultdict(list)]
    for t,pair in enumerate(raw['actions']):
        obs=sim.obs(s);own=obs['players'][s]
        for uid,w in own[2].items():
            if w[2]=='COLLECTOR':jobs[0].append((uid,t,freight(obs,cfg,uid,pair[s].get(uid))))
        for tree in own[3]:jobs[1].append((tree,t,tree_features(obs,tree,cfg)))
        sim.cash.events.clear();sim.step(*pair)
        for kind,side,v,*arg in sim.cash.events:
            if kind=='own_delivery' and side==s:flows[0][arg[0]].append((t,v))
            if kind=='tree_absorption':flows[1][arg[0]].append((t,v))
    assert sim.cash==raw['official_score_by_seat']
    return [([x for _,_,x in records],[sum(v for at,v in flow[uid] if t<=at<t+HORIZON) for uid,t,_ in records])
            for records,flow in zip(jobs,flows)]

def run():
    games=sr.load(ACTOR/'tactical_exchange_dev/paired.json.gz')
    data=[collect(g) for g in games];model={};checks=[]
    for j,k in enumerate(('freight','tree')):
        subsets=lambda filt:tuple(np.concatenate([data[i][j][n] for i,g in enumerate(games) if filt(g)]) for n in (0,1))
        x,y=subsets(lambda g:g['seed']<=621758326);diag=np.eye(x.shape[1])*30;diag[0,0]=0
        w=np.linalg.solve(x.T@x+diag,x.T@y);model[k]=w.tolist()
        for seed in (621758327,621758328):
            test,truth=subsets(lambda g:g['seed']==seed)
            pr=np.maximum(0,test@w)
            checks.append(dict(target=k,seed=seed,train_count=len(y),test_count=len(truth),mae=float(np.abs(pr-truth).mean()),mean_train_baseline_mae=float(np.abs(truth-np.mean(y)).mean())))
    model['threat']=sr.load(ACTOR/'tree_screen_response_dev/planned_threat_model.json')['w']
    sr.retain(HERE/'weights.json',model);sr.retain(HERE/'metrics.json',checks)
    print(checks)
if __name__=='__main__':run()
