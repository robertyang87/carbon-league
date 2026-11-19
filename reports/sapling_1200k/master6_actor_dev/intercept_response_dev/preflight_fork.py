"""Use the original's already issued simultaneous order to score the historical changed destination."""
from match import HERE,ACTOR,native
from forecast import cargo_map,carriers,_DIRECTION,_TO
import numpy as np

def trace(r,weights):
    path=ACTOR/'planter_capture_full_dev/games/candidate'/f"raw_{r['seed']}_{r['seat']}.json.gz"
    assert native.sha(path)==r['trajectory_sha256']
    raw=native.read(path);s=r['seat']
    sim=native.exam.FastCarbon(native.exam.CFG).reset(seed=raw['seed'])
    target=r['changes'][0];t=target['step']
    for pair in raw['actions'][:t]:sim.step(*pair)
    pair=raw['actions'][t];old,new=[dict(v) for v in pair],[dict(v) for v in pair]
    for k,(before,after) in target['units'].items():
        assert old[s].get(k)==before
        if after:new[s][k]=after
        else:new[s].pop(k,None)
    forecast=[];obs=sim.obs(s);field=cargo_map(obs,weights)
    for uid,load,cells,x in carriers(obs['players'],1-s):
        l=x@weights;pr=np.exp(l-l.max());pr/=pr.sum()
        y=_DIRECTION.index(pair[1-s].get(uid) or None)
        forecast.append(dict(uid=uid,load=load,destinations=cells.tolist(),probabilities=pr.tolist(),actual=int(cells[y])))
    branches=[]
    for orders in (old,new):
        fork=native.exam.FastCarbon(native.exam.CFG).set_state(sim.get_state());fork.step(*orders)
        branches.append(dict(cash=fork.cash,collision=fork.last_collisions,
                workers={k:None if k not in fork.workers else [fork.workers[k].pos,fork.workers[k].carbon] for k in target['units']}))
    return dict(seed=raw['seed'],seat=s,step=t,forecast=forecast,branches=branches)
if __name__=='__main__':
    rows=[trace(r,np.load(HERE/'weights.npz')['weights']) for r in native.read(HERE/'preflight.json') if r['changes']]
    native.save(HERE/'preflight_fork.json',rows)
