"""Select public-state strata under split, seat and time balance, without looking at actions, scores or yields."""
from collections import Counter,defaultdict,deque
from pathlib import Path
import sys
from register import HERE,ROOT,BASE,sha,read,stable,seal
from policy import snapshot
sys.path.insert(0,str(ROOT/'code'))
from fastsim import FastCarbon

def select(rows,n,per_state):
    rows=sorted(rows,key=lambda r:stable("phase-switch-frozen-roots",r["id"]))
    # Minimize any split imbalance first, then choose the deterministic feasible assignment.
    groups=defaultdict(list)
    for r in rows:groups[r["map"]].append(r)
    maps=sorted(groups,key=lambda m:stable("frozen-source-order",m))
    def search(i,selected,p,seat0,early):
        if len(selected)==n:
            if p!=per_state or seat0!=n//2 or early!=n//2:return None
            if any(len({c[k] for c in selected if c['regime']==g})!=2 for k in ('t','seat') for g in ('stable','pressure')):return None
            return selected
        required=n-len(selected)
        if len(maps)-i<required or p>per_state or seat0>n//2 or early>n//2:return None
        if p+required<per_state or seat0+required<n//2 or early+required<n//2:return None
        for r in groups[maps[i]]:
            v=search(i+1,selected+[r],p+(r["regime"]=="pressure"),seat0+(r["seat"]==0),early+(r["t"]==195))
            if v is not None:return v
        return search(i+1,selected,p,seat0,early)
    return search(0,[],0,0,0)

def main():
    plan=read(HERE/'protocol.json');assert sha(HERE/'policy.py')==plan['frozen_policy_sha256']
    catalog=[]
    for g in plan["eligible"]:
        p=ROOT/g["raw"];assert sha(p)==g["raw_sha256"]
        raw=read(p);cfg=raw['game_configuration'];sim=FastCarbon(cfg).reset(seed=g['seed']);hist=deque(maxlen=13)
        for t,joint in enumerate(raw["actions"][:226]):
            obs=sim.obs(g['seat'])
            if t in (195,225):
                info=snapshot(obs,list(hist))
                row={**g,"id":f"m{g['seed']}_s{g['seat']}_t{t}","t":t,"regime":info["regime"],"state":info}
                row["lookback"]={str(n):{"relative_cash":hist[-n]['players'][g['seat']][0]-hist[-n]['players'][1-g['seat']][0],
                    "worker_amounts":[len(side[2]) for side in hist[-n]['players']]} for n in (3,5,10)}
                catalog.append(row)
            hist.append(obs)
            if t<225:sim.step(*joint)
    train=select([r for r in catalog if r["split"]=="train"],6,3)
    held=select([r for r in catalog if r["split"]=="held"],4,2)
    census=Counter((r["split"],r["regime"],r["seat"],r["t"]) for r in catalog)
    seal(HERE/"census.json.gz",{"protocol_sha256":sha(HERE/"protocol.json"),"eligible_roots":catalog,
        "public_strata":{"/".join(map(str,k)):v for k,v in census.items()},"train_support":train is not None,"held_support":held is not None})
    if train is None or held is None:print("NO SUPPORT",bool(train),bool(held));return
    chosen=train+held
    for g in ('pressure','stable'):
        next(r for r in train if r['regime']==g)['repeat']=True
    for r in chosen:r.setdefault('repeat',False)
    seal(HERE/"roots.json",{"protocol_sha256":sha(HERE/"protocol.json"),"census_sha256":sha(HERE/"census.json.gz"),
        "roots":chosen,"raw_original_jobs":0,"pair_unit_map":True})
    print([(r['id'],r['regime'],r['split'],r['repeat'])for r in chosen])
if __name__=="__main__":main()
