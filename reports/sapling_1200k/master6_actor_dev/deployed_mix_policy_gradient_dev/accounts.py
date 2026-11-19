"""Attribute full-game changes to both sides' actual cash writes; retain every distinct map."""
from collections import Counter
import sys
from study import HERE,PLAN,arena,World
sys.path.insert(0,str(HERE.parents[1]))
from turning_point_ledger import CashEvents

def run():
    cases=arena.read(HERE/'development.json')['rows'];out=[]
    for case in cases:
        seed=case['seed'];s=case['seat'];name=case['arm']
        src=HERE/'games'/f'{seed}_{s}_{name}.json.gz';raw=arena.read(src);w=World(seed);led=[Counter(),Counter()]
        for pair in raw['actions']:
            w._sim.cash=CashEvents(w._sim.cash);w.advance(pair)
            for kind,owner,v,*extra in w._sim.cash.events:led[owner][kind]+=v
        if any(abs(w._sim.cash[p]-raw['score'][p])>.011 or abs(80+sum(led[p].values())-raw['score'][p])>.011 for p in (0,1)):
            raise ValueError('ledger does not close')
        out.append(dict(seed=seed,seat=s,arm=name,accounts=[dict(led[i]) for i in (s,1-s)],source=arena.sha(src)))
    diffs=[]
    for seed in PLAN['development_seeds']:
        pair={x['arm']:x for x in out if x['seed']==seed and x['seat']==0}
        d=[]
        for mine in (0,1):
            a,b=[e['accounts'][mine] for e in (pair['parent'],pair['candidate'])]
            d.append({k:b.get(k,0)-a.get(k,0) for k in sorted(a.keys()|b.keys())})
        diffs.append(dict(seed=seed,seat0_own_opponent=d))
    arena.write(HERE/'development_accounts.json',dict(cases=out,differences=diffs))
if __name__=='__main__':run()
