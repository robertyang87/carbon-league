"""Reconcile active collector counts into births, deaths, payroll and open safe hiring slots."""
from collections import Counter
from scan import sr,HERE
from evade import _D,_TO,_DIRECTION
from turning_point_ledger import CashEvents
def measure(c):
    raw=sr.load(sr.ROOT/c['source']);sim=sr.FastCarbon(raw['game_configuration']).reset(seed=c['seed'])
    sim.cash=CashEvents(sim.cash);out=[];slot=[Counter(),Counter()]
    for t,pair in enumerate(raw['actions']):
        before=[{k for k,w in sim.workers.items() if w.player==s and w.is_collector} for s in (0,1)]
        for s in (0,1):
            own,enemy=sim.obs(s)['players'][s],sim.obs(s)['players'][1-s]
            hires=sum(pair[s].get(k) in ('RECCOLLECTOR','RECPLANTER') for k in own[1])
            used={_TO[w[0],_DIRECTION.index(pair[s].get(k))] for k,w in own[2].items()}
            free=[p for k,p in own[1].items() if not pair[s].get(k) and p not in used and all(_D[p,w[0]]>1 for w in enemy[2].values())]
            cfg=raw['game_configuration']
            price=cfg['recCollectorCost']+sum(map(len,before))*cfg['recCollectorIncreaseCost']
            if hires==0 and len(own[2])<cfg['workerLimit'] and own[0]>=price and free:
                slot[s]['safe_idle_slot_turns']+=1
            slot[s]['active_collector_turns']+=len(before[s])
            slot[s]['other_worker_turns']+=len(own[2])-len(before[s])
        sim.cash.events.clear();sim.step(*pair)
        for s in (0,1):
            after={k for k,w in sim.workers.items() if w.player==s and w.is_collector}
            born=sum(e[0]=='hire_collector' and e[1]==s for e in sim.cash.events)
            slot[s]['paid_hires']+=born;slot[s]['deaths']+=len(before[s])+born-len(after)
            slot[s]['births_surviving_turn']+=len(after-before[s])
        if t+1 in (50,100,150,200,250,len(raw['actions'])):
            out.append(dict(end=t+1,own=slot[c['seat']],opponent=slot[1-c['seat']]));slot=[Counter(),Counter()]
    return dict(seed=c['seed'],seat=c['seat'],phases=out)
if __name__=='__main__':
    rows=[measure(c) for c in sr.load(HERE/'coverage.json.gz')];sr.retain(HERE/'staffing.json.gz',rows)
    for i in range(6):
        print(rows[0]['phases'][i]['end'],[dict(sum((Counter(c['phases'][i][s]) for c in rows),Counter())) for s in ('own','opponent')])
