"""Trace the largest ten-turn growth of the examiner lead back to public positions and both players' commands."""
from collections import Counter
from measure import HERE,sr
from turning_point_ledger import CashEvents

def check(path,seat):
    raw=sr.load(path);sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed']);sim.cash=CashEvents(sim.cash)
    leads=[sim.cash[1-seat]-sim.cash[seat]];rows=[]
    for t,pair in enumerate(raw['actions']):
        obs=sim.obs(seat);public=obs['players']
        sim.cash.events.clear();sim.step(*pair)
        rows.append(dict(t=t,players=public,orders=pair,receipts=list(sim.cash.events),collisions=sim.last_collisions))
        leads.append(sim.cash[1-seat]-sim.cash[seat])
    width=min(len(rows),10);start=max(range(len(rows)-width+1),key=lambda t:leads[t+width]-leads[t]);end=start+width
    comp=Counter();big=[]
    for r in rows[start:end]:
        for k,s,v,*d in r['receipts']:
            comp[k]+=v*(1 if s!=seat else -1)
            if k=='own_delivery':big.append(dict(t=r['t'],seat=s,amount=v,worker=d[0]))
    assert abs(sum(comp.values())-(leads[end]-leads[start]))<1e-5
    return dict(seat=seat,arm=path.parent.name,seed=raw['seed'],sha256=sr.sha(path),
           start=start,end=end,lead=[leads[start],leads[end]],components=dict(comp),deliveries=big,
           public_prefix={str(k):rows[max(0,start-k)] for k in [10,5,3,0]},through_window=rows[start:end])
if __name__=='__main__':
    import sys
    arm=sys.argv[1];seed=sr.load(HERE/'protocol.json')['seeds'][0]
    rows=[check(HERE/f'games/{arm}/raw_{seed}_{s}.json.gz',s) for s in (0,1)]
    sr.retain(HERE/f'turning_points_{arm}.json.gz',rows)
    for r in rows:print(r['seat'],r['start'],r['end'],r['lead'],sorted(r['deliveries'],key=lambda x:-x['amount'])[:2])
