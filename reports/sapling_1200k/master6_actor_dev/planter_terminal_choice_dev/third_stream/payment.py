"""Predict realized immediate purchases from pre-action public states; fit and evaluate by map."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parents[1]
sys.path.insert(0,str(ACTOR.parent/'all_top5'))
import fetch

NAMES=['adjacent_collectors','adjacent_planters','second_ring_collectors','second_ring_planters',
       'adjacent_collector_load','adjacent_enemy_approaching','adjacent_friendly','distance_own_base',
       'distance_enemy_base','log_cash_over_price','fraction_left']
def distance(a,b):
    y,x=abs(a//15-b//15),abs(a%15-b%15)
    return min(y,15-y)+min(x,15-x)

def encode(players,s,pos,previous,t,price):
    own,enemy=players[s],players[1-s]
    d=[0.,0.,0.,0.];load=closing=0.
    for uid,(p,cargo,kind) in enemy[2].items():
        gap=distance(p,pos)
        if gap in (1,2):d[2*(gap-1)+int(kind=='PLANTER')]+=1
        if gap==1:
            if kind=='COLLECTOR':load+=np.log1p(cargo)/6
            if uid in previous and distance(previous[uid][0],pos)>gap:closing+=1
    friendly=sum(distance(pos,w[0])==1 for w in own[2].values())
    return [*d,load,closing,friendly,min(distance(pos,p) for p in own[1].values())/14,
            min(distance(pos,p) for p in enemy[1].values())/14,np.log1p(own[0]/price),(299-t)/299]

def extract(row,s,data):
    env,_,_=fetch.extract(data);steps=env['steps'];rows=[]
    for t in range(len(steps)-1):
        p=steps[t][0]['observation']['players'];own=p[s]
        if not own[2]:continue
        occupied={x for side in p for x in side[1].values()}|{v[0] for side in p for v in side[3].values()}
        price=20+5*1.235**sum(v[1]<50 for side in p for v in side[3].values()) if own[3] else 20.
        if own[0]<price:continue
        after=steps[t+1][0]['observation'];orders=steps[t+1][s].get('action') or {}
        purchases={pos:tid for tid,(pos,age) in after['players'][s][3].items() if age==1}
        previous=steps[t-1][0]['observation']['players'][1-s][2] if t else {}
        for wid,(pos,_,kind) in own[2].items():
            if kind!='PLANTER' or pos in occupied or orders.get(wid) not in (None,'NONE'):continue
            x=encode(p,s,pos,previous,t,price)
            tid=purchases.get(pos);y=int(bool(tid and after['trees'][tid][0]==wid))
            rows.append((x,y,row['seed']))
    return rows

def prob(z):
    return 1/(1+np.exp(-np.clip(z,-35,35)))

def main():
    top=ACTOR.parent/'all_top5';records={}
    for path in [top/'manifest.json',*sorted(top.glob('appendix_*/manifest.json'))]:
        for r in json.loads(path.read_bytes())['games']:
            if r.get('frame_verified') and 'claude-AlphaCarbon-Master6' in r['seat_names']:records[r['id']]=r
    rows=[]
    for key,r in sorted(records.items()):
        data=(fetch.CACHE/'replays'/(key+'.json.gz')).read_bytes()
        assert fetch.sha(data)==r['sha256']
        for s,name in enumerate(r['seat_names']):
            if name=='claude-AlphaCarbon-Master6':rows.extend(extract(r,s,data))
    X=np.asarray([v[0] for v in rows]);y=np.asarray([v[1] for v in rows]);seeds=np.asarray([v[2] for v in rows])
    held=np.asarray([hashlib.sha256(str(s).encode()).digest()[0]%5==0 for s in seeds]);train=~held
    mu=X[train].mean(0);sd=X[train].std(0);sd[sd<1e-8]=1
    z=np.column_stack((np.ones(len(X)),(X-mu)/sd))
    penalty=np.diag([0]+[12]*X.shape[1]);beta=np.zeros(z.shape[1])
    for _ in range(40):
        p=prob(z[train]@beta)
        grad=z[train].T@(p-y[train])+penalty@beta
        hess=(z[train].T*(p*(1-p)))@z[train]+penalty
        change=np.linalg.solve(hess,grad);beta-=change
        if np.max(abs(change))<1e-8:break
    keys=(X[:,0]>0).astype(int)+2*(X[:,1]>0)
    rates=np.array([(1+y[train&(keys==k)].sum())/(2+sum(train&(keys==k))) for k in range(4)])
    def evaluate(mask):
        return dict(n=int(sum(mask)),maps=int(len(set(seeds[mask]))),purchases=int(sum(y[mask])),
                    brier={k:float(np.mean((p-y[mask])**2)) for k,p in
                           [('global',y[train].mean()),('type',rates[keys[mask]]),('head',prob(z[mask]@beta))]})
    np.savez_compressed(HERE/'payment.npz',features=NAMES,mean=mu,scale=sd,beta=beta,type_rates=rates)
    report=dict(roster=len(records),train=evaluate(train),held=evaluate(held),held_threat=evaluate(held&(keys>0)),
                held_maps=sorted({int(s) for s in seeds[held]}))
    (HERE/'payment_result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['held'],report['held_threat'])
if __name__=='__main__':main()
