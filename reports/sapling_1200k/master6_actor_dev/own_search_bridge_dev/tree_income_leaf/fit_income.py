"""Fit eight-step retained tree income on signed matches and score real continuation choices."""
from functools import partial
import numpy as np
import check

b = check.bridge
OUT = check.HERE
def features(sim,t):
    horizon=max(0,min(8,sim.config['episodeSteps']-1-sim.step_count,sim.config['treeLifespan']-t.age))
    receipt=sim.trees_dict[t.id][1]
    enemy_dist=min((b.F.DIST[t.pos,w.pos] for w in sim.workers.values()
                    if w.player!=t.player and not w.is_collector),default=15)
    competing=sum(o.id!=t.id and b.F.DIST[t.pos,o.pos]<=2 for o in sim.trees.values())
    h=horizon/8
    return np.array([h,h*receipt,h*np.sqrt(receipt),h*min(4,enemy_dist)/4,
                     h*receipt*(enemy_dist<=2),h*min(4,competing)/4,h*(t.age<=1)])

def examples(rec):
    p=b.ROOT/rec['raw']
    assert b.sha(p)==rec['raw_sha256']
    raw=b.load(p)
    sim=b.FastCarbon(raw['game_configuration']).reset(seed=rec['seed'])
    frames=[]
    for pair in raw['actions']:
        sim.step(*pair)
        frames.append({t.id:(t.player,sim.trees_dict[t.id][1],features(sim,t))
                       for t in sim.trees.values()})
    assert list(sim.cash)==raw['last_post_cash']
    rows=[]
    for i,frame in enumerate(frames[:-1]):
        for tid,(owner,_,x) in frame.items():
            target=0.
            for future in frames[i+1:i+9]:
                if tid not in future or future[tid][0]!=owner:break
                target+=future[tid][1]
            rows.append((x,target))
    return rows

def learned_leaf(w,sim,seat,pre):
    value,end=check.base_leaf(sim,seat,pre)
    if end:return value,True
    for t in sim.trees.values():
        amount=max(0.,float(features(sim,t)@w))
        value+=amount if seat==t.player else -amount
    return value,False

def main():
    recs=b.load(OUT.parent/'fit.json')['trajectories']
    fit={r['terrain'] for r in recs if r['split']=='fit'}
    hold={r['terrain'] for r in recs if r['split']!='fit'}
    assert len(fit)==6 and len(hold)==2 and not fit & hold
    data={'fit':[],'hold':[]}
    for r in recs:data['fit' if r['terrain'] in fit else 'hold'].extend(examples(r))
    x,y=zip(*data['fit']);x=np.asarray(x);y=np.asarray(y)
    w=np.linalg.solve(x.T@x/len(y)+.01*np.eye(x.shape[1]),x.T@y/len(y))
    x,y=zip(*data['hold']);x=np.asarray(x);y=np.asarray(y)
    rate=b.load(b.ROOT/recs[0]['raw'])['game_configuration']['cellAbsorptionRate']
    turns=np.rint(x[:,0]*8).astype(int)
    receipts=np.divide(x[:,1],x[:,0],out=np.zeros_like(x[:,0]),where=x[:,0]>0)
    simple=receipts*sum((turns>=i)*(1-rate)**i for i in range(1,9))
    branches={r['terrain'] for r in b.load(OUT.parent/'outcome_audit.json')['roots']}
    assert not (fit|hold) & branches
    b.save(OUT/'learned.json',dict(weights=w.tolist(),train_trees=len(data['fit']),hold_trees=len(y),
           hold_mae=float(np.abs(np.maximum(0,x@w)-y).mean()),zero_mae=float(np.abs(y).mean()),simple_mae=float(np.abs(simple-y).mean())))
    check.run(partial(learned_leaf,w),OUT/'learned_results.json')
if __name__=='__main__':main()
