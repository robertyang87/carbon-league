"""Trace the next actual delivery or death of each marked worker in the source game."""
import sys
from audit import sr, HERE
from turning_point_ledger import CashEvents

def trace():
    rows=[]
    for case in sr.load(HERE/'audit.json'):
        raw=sr.load(sr.ROOT/case['source']); seat=case['seat']
        hits={r['t']:r for r in case['hits']}
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=case['seed'])
        sim.cash=CashEvents(sim.cash)
        pending=[]
        for t,pair in enumerate(raw['actions']):
            if t in hits:
                row=dict(seed=case['seed'], seat=seat, **hits[t])
                rows.append(row);pending.append(row)
            sim.cash.events.clear();sim.step(*pair)
            for r in pending[:]:
                uid=r['worker']
                payment=sum(e[2] for e in sim.cash.events if e[0]=='own_delivery' and e[1]==seat and e[3]==uid)
                if payment or uid not in sim.workers or t==len(raw['actions'])-1:
                    r.update(resolution=t,delivery=payment,alive=uid in sim.workers)
                    pending.remove(r)
    sr.retain(HERE/'historical_settlement.json', rows)
    print(len(rows),sum(r['delivery']>0 for r in rows))
if __name__=='__main__':trace()
