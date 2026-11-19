"""Count public forecast opportunities before neural action scoring, without using opponent commands."""
from collections import Counter
from match import HERE,ACTOR,native
from forecast import cargo_map,carriers,_TO,_D
import numpy as np

def measure(path,weights):
    raw=native.read(path);s=raw['candidate_seat'];sim=native.exam.FastCarbon(native.exam.CFG).reset(seed=raw['seed'])
    c=Counter()
    for pair in raw['actions']:
        obs=sim.obs(s);p=obs['players'];c['turns']+=1
        preds=list(carriers(p,1-s))
        if preds:
            c['forecast_turns']+=1
            field=cargo_map(obs,weights)
            for uid,(pos,load,kind) in p[s][2].items():
                if kind!='COLLECTOR' or load>=60:continue
                dest=_TO[pos]
                rivals=[w[0] for w in p[1-s][2].values() if w[2]=='COLLECTOR' and w[1]<=load]
                safe=np.array([cell not in p[1-s][1].values() and all(_D[cell,other]>1 for other in rivals) for cell in dest])
                # Before resolving ally reservations in the autoregressive decoder.
                reach=field[dest]*safe
                if reach.max()>0:c['unit_turns_with_mass']+=1
                if reach.max()>=.25:c['unit_turns_above_quarter']+=1
                if reach.max()>=.5:c['unit_turns_above_half']+=1
        sim.step(*pair)
    return dict(seed=raw['seed'],seat=s,counts=dict(c))
if __name__=='__main__':
    w=np.load(HERE/'weights.npz')['weights']
    dirs=[ACTOR/'capture_reach_dev/games/parent',ACTOR/'planter_capture_full_dev/games/candidate']
    rows=[measure(p,w) for d in dirs for p in sorted(d.glob('raw_*.json.gz'))]
    native.save(HERE/'coverage.json',rows)
    print([r['counts'] for r in rows])
