"""Account for each focal worker from the first real intervention to the official finish."""
from rescue import HERE
import sys
sys.path.insert(0,str(HERE.parent/'bank_under_pressure_dev'))
from audit import sr
from turning_point_ledger import CashEvents
def verify():
    evidence=[]
    for r in sr.load(HERE/'full_ledger.json'):
        if not r['edits']:continue
        seat=r['seat'];first=r['edits'][0];uid=first['worker']
        path=HERE/'games/candidate'/f"raw_{r['seed']}_{seat}.json.gz";raw=sr.load(path)
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=r['seed']);sim.cash=CashEvents(sim.cash)
        log=[];before=None
        for t,pair in enumerate(raw['actions']):
            if t==first['t']:before=list(sim.cash)
            sim.cash.events.clear();sim.step(*pair)
            if t<first['t']:continue
            w=sim.workers.get(uid);log.append(dict(t=t,order=pair[seat].get(uid),pos=w.pos if w else None,load=w.carbon if w else None,
                receipts=[e for e in sim.cash.events if e[0]=='own_delivery' and e[3]==uid],collision=[e for e in sim.last_collisions if e[0]==uid]))
        evidence.append(dict(seed=r['seed'],seat=seat,worker=uid,initial=first,margin_before=before[seat]-before[1-seat],final=r['margin'],log=log))
    sr.retain(HERE/'focal_receipts.json',evidence)
    for x in evidence:print(x['seat'],x['margin_before'],x['initial']['cargo'],[e[2] for v in x['log'] for e in v['receipts']])
if __name__=='__main__':verify()
