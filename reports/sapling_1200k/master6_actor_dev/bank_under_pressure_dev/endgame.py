"""Inventory and subsequent actual bank receipts for both sides' late-game carriers."""
from audit import HERE,sr
from bank import _D
from turning_point_ledger import CashEvents
from collections import Counter
def measure(source,seat):
    raw=sr.load(source);sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
    sim.cash=CashEvents(sim.cash)
    census=[];events=[];cash=[]
    for t,pair in enumerate(raw['actions']):
        cash.append(list(sim.cash))
        if t in (240,270,285,295):
            ob=sim.obs(seat);carriers=[]
            for side,player in enumerate(ob['players']):
                for uid,(pos,load,kind) in player[2].items():
                    if kind=='COLLECTOR' and load>=1:
                        carriers.append(dict(uid=uid,side=side,load=load,home=int(min(_D[pos,b] for b in player[1].values()))))
            census.append(dict(t=t,cash=list(sim.cash),carriers=carriers))
        sim.cash.events.clear();sim.step(*pair);events.append(list(sim.cash.events))
    for c in census:
        flows=[Counter(),Counter()];first={}
        for t in range(c['t'],len(events)):
            for k,s,v,*meta in events[t]:
                flows[s][k]+=v
                if k=='own_delivery':first.setdefault(meta[0],(t,v))
        for w in c['carriers']:
            w['next_bank']=list(first[w['uid']]) if w['uid'] in first else None
        c['future_cash_flows']=flows
    return dict(seed=raw['seed'],seat=seat,source=str(source.relative_to(sr.ROOT)),census=census)
if __name__=='__main__':
    historical=[(sr.ROOT/c['source'],c['seat']) for c in sr.load(HERE/'audit.json')]
    sr.retain(HERE/'historical_endgame.json.gz',[measure(p,s) for p,s in historical])
    if (HERE/'scores.json').exists():
        sr.retain(HERE/'full_endgame.json.gz',[measure(HERE/'games'/r['arm']/f"raw_{r['seed']}_{r['seat']}.json.gz",r['seat']) for r in sr.load(HERE/'scores.json')])
