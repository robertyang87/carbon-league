"""Audit genuine opponent responses to each changed joint action, plus total receipts in both arms."""
from match import HERE,native
from forecast import carriers,_DIRECTION,_TO
import numpy as np

def destination(pos,cmd):
    return int(_TO[pos,0 if not cmd or cmd=='NONE' else _DIRECTION.index(cmd)])

def audit(path,changes):
    raw=native.read(path);s=raw['candidate_seat'];sim=native.exam.FastCarbon(native.exam.CFG).reset(seed=raw['seed'])
    weights=np.load(HERE/'weights.npz')['weights']
    events=[];totals=[dict(won_in_collision=0.,normal_deposit=0.,tree_income=0.) for _ in range(2)]
    from turning_point_ledger import CashEvents
    sim.cash=CashEvents(sim.cash)
    for t,pair in enumerate(raw['actions']):
        start=sim.obs(s)['players']
        current=[]
        if t in changes:
            # Both orders were submitted simultaneously. Read the opponent command only for the audit.
            for uid,orders in changes[t].items():
                if uid not in sim.workers or not sim.workers[uid].is_collector:continue
                w=sim.workers[uid]
                before,after=[destination(w.pos,cmd) for cmd in orders]
                targets=[]
                for eid,load,sites,x in carriers(start,1-s):
                    logits=x@weights;p=np.exp(logits-logits.max());p/=p.sum()
                    if before not in sites and after not in sites:continue
                    target=sim.workers[eid]
                    realized=destination(target.pos,pair[1-s].get(eid))
                    targets.append(dict(id=eid,load=load,p_old=float(p[sites==before].sum()),
                                        p_new=float(p[sites==after].sum()),actual_dest=realized,
                                        met_on_old_order=realized==before,met_on_new_order=realized==after))
                current.append(dict(step=t,unit=uid,load=w.carbon,old_cell=before,new_cell=after,targets=targets))
        sim.cash.events.clear();sim.step(*pair)
        for eid,loser,load,payee,winner in sim.last_collisions:
            if winner is not None and winner!=loser:totals[winner]['won_in_collision']+=load
        for cat,seat,val,*extra in sim.cash.events:
            if cat=='own_delivery':totals[seat]['normal_deposit']+=val
            elif cat=='tree_absorption':totals[seat]['tree_income']+=val
        for e in current:
            e.update(alive_after=e['unit'] in sim.workers,
                     collisions=[v for v in sim.last_collisions if any(v[0]==z['id'] for z in e['targets'])])
        events.extend(current)
    assert list(sim.cash)==raw['official_score_by_seat']
    return dict(arm=path.parent.name,seat=s,score=raw['official_score_by_seat'],totals=totals,events=events)
if __name__=='__main__':
    rows=native.read(HERE/'changes.json')
    out=[]
    for path in sorted((HERE/'games').glob('*/raw_*.json.gz')):
        raw=native.read(path);s=raw['candidate_seat']
        changes={r['step']:r['units'] for r in rows[s]['changes']} if path.parent.name=='candidate' else {}
        out.append(audit(path,changes))
    assert len(out)==4
    native.save(HERE/'intercepts.json',out)
