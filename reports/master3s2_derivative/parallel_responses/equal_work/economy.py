"""Reconcile the independent response labels with the official cash flow, by seat and by carrier."""
from collections import Counter
import copy
import gzip
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from panel import PLAN,sha,sources,put
from immediate import econ

def run():
    rdoc=json.loads((HERE/'protocol.json').read_bytes())
    rows=[]
    for chosen in rdoc['selected']:
        seed,seat,t=chosen['seed'],chosen['seat'],chosen['step']
        card=next(c for c in PLAN['tickets'] if c['seed']==seed and c['seat']==seat)
        _,turns,frames=sources(card)
        s=econ.FastCarbon(econ.config()).reset(seed=seed)
        for i in range(t):
            assert s.obs(seat)['players']==frames[i]['observations'][seat]['players']
            s.step(*(c['response']['action'] for c in turns[i]['calls']))
        assert s.obs(seat)['players']==frames[t]['observations'][seat]['players']
        own=s.obs(seat)['players'][seat]
        orders=next(r for r in rdoc['cohort'] if (r['seed'],r['seat'],r['step'])==(seed,seat,t))
        changed={u:{'observed':own[2].get(u,[own[1].get(u),'BASE']),'single0':orders['single0'].get(u,'NONE'),
                    'mean2':orders['mean2'].get(u,'NONE')} for u in sorted(orders['single0'].keys()|orders['mean2'].keys()) if orders['single0'].get(u)!=orders['mean2'].get(u)}
        parts={}
        for name in ('single0','mean2'):
            raw=(HERE/'branches'/f'{seed}_{seat}_{name}.json.gz').read_bytes()
            doc=json.loads(gzip.decompress(raw));sim=copy.deepcopy(s);sim.cash=econ.CashEvents(sim.cash)
            summary=[Counter(),Counter()];itinerary=[]
            for r in doc['tape']:
                sim.cash.events.clear();sim.step(*r['orders'])
                assert list(sim.cash)==r['end_cash']
                for k,count in enumerate(econ.cash_flow(sim.cash.events)):summary[k].update(count)
                itinerary.extend(dict(step=i+1,side=p,amount=round(v,4),worker=more[0]) for i in [r['step']]
                                 for kind,p,v,*more in sim.cash.events if kind=='own_delivery')
            assert list(sim.cash)==doc['scores']
            parts[name]=dict(source_sha256=sha(raw),by_seat=summary,deliveries=itinerary,score=doc['scores'])
        flows=[]
        for k in (0,1):
            a=parts['single0']['by_seat'][k];b=parts['mean2']['by_seat'][k]
            flows.append({tag:round(b.get(tag,0)-a.get(tag,0),4) for tag in sorted(a.keys()|b.keys())})
        effects=[round(y-x,2) for x,y in zip(parts['single0']['score'],parts['mean2']['score'])]
        assert all(abs(sum(flow.values())-effect)<.001 for flow,effect in zip(flows,effects))
        rows.append(dict(seed=seed,seat=seat,step=t,root_changed=changed,arms=parts,flow_delta_by_physical_seat=flows,
                         final_cash_delta_by_physical_seat=effects,change_in_own_margin=round(effects[seat]-effects[1-seat],2)))
    put(HERE/'economy.json',rows)
    for r in rows:print(r['seed'],r['root_changed'],r['final_cash_delta_by_physical_seat'],r['flow_delta_by_physical_seat'])
if __name__=='__main__':run()
