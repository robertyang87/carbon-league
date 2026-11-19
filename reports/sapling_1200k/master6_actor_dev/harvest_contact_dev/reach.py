"""Upper bound on changed immediate captures for a single available friendly collector on recorded orders."""
from scan import HERE,sr
from evade import situations,_D,_TO,_DIRECTION
def tally():
    rows=[]
    for c in sr.load(HERE/'screen.json'):
        raw=sr.load(sr.ROOT/c['source']);s=c['seat'];sim=sr.FastCarbon(raw['game_configuration']).reset(seed=c['seed']);prev=None
        for t,pair in enumerate(raw['actions'][:201]):
            obs=sim.obs(s);own=obs['players'][s];enemy=obs['players'][1-s]
            for target,load,cells,x in list(situations(obs,prev)) if t>=50 else []:
                actual=int(cells[_DIRECTION.index(pair[1-s].get(target))])
                planned={k:int(_TO[w[0],_DIRECTION.index(pair[s].get(k))]) for k,w in own[2].items()}
                reachable=[];safe=[]
                for uid,(p,cargo,kind) in own[2].items():
                    if kind!='COLLECTOR' or cargo>=load or _D[p,actual]>1 or actual==planned[uid]:continue
                    reachable.append(uid)
                    if cargo>25 or actual in enemy[1].values() or any(k!=uid and z==actual for k,z in planned.items()):continue
                    if any(w[2]=='COLLECTOR' and w[1]<=cargo and _D[w[0],actual]<=1 for w in enemy[2].values()):continue
                    if any(b==actual and pair[s].get(k) for k,b in own[1].items()):continue
                    safe.append(uid)
                rows.append(dict(seed=c['seed'],seat=s,t=t,target=target,load=load,actual=actual,possible=reachable,safe=safe))
            sim.step(*pair);prev=obs
    sr.retain(HERE/'reach.json',rows)
    print(len(rows),sum(bool(r['possible']) for r in rows),sum(bool(r['safe']) for r in rows))
if __name__=='__main__':tally()
