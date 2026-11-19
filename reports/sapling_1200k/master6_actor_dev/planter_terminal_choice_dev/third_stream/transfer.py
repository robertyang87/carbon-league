"""Run the frozen payment head on recorded full native matches without re-calling either actor."""
from collections import Counter
import json
import numpy as np
from payment import ACTOR,HERE,encode,prob
from run import retest
sr=retest.sr

def inputs(p,s):
    own=p[s]
    occupied=set().union(*(side[1].values() for side in p),*({v[0] for v in side[3].values()} for side in p))
    price=20+5*1.235**sum(v[1]<50 for side in p for v in side[3].values()) if own[3] else 20.
    return occupied,price

def predict(w,x):
    key=int(x[0]>0)+2*int(x[1]>0)
    z=(np.asarray(x)-w['mean'])/w['scale']
    return float(prob(w['beta'][0]+z@w['beta'][1:])),float(w['type_rates'][key])

def main():
    rows=[];scores=[]
    with np.load(HERE/'payment.npz') as w:
        for dirname in ('planter_escape_full_dev','planter_payback_full_dev','planter_capture_full_dev','capture_reach_dev'):
            for path in sorted((ACTOR/dirname/'games').glob('*/raw_*.json.gz')):
                raw=sr.load(path);s=raw['candidate_seat']
                sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed']);prev={}
                for t,pair in enumerate(raw['actions']):
                    p=sim.obs(s)['players'];occupied,price=inputs(p,s);own=p[s]
                    pending=[]
                    if own[0]>=price:
                        for uid,(pos,_,kind) in own[2].items():
                            if kind=='PLANTER' and pos not in occupied and not pair[s].get(uid):
                                x=encode(p,s,pos,prev,t,price)
                                pending.append((uid,pos,predict(w,x),bool(x[0] or x[1])))
                    prev=p[1-s][2];sim.step(*pair)
                    paid={v.pos:sim.trees_dict[k][0] for k,v in sim.trees.items() if v.age==1 and v.player==s}
                    for uid,pos,(head,base),threat in pending:
                        rows.append(dict(game=str(path.relative_to(ACTOR)),step=t,unit=uid,seed=raw['seed'],
                                         head=head,type_rate=base,threat=threat,bought=paid.get(pos)==uid))
                assert list(sim.cash)==raw['official_score_by_seat']
        for s in (0,1):
            q=sr.load(HERE/f'seat{s}/live_protocol.json');raw=sr.load(sr.ROOT/q['record']['source'])
            sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed']);prev={}
            for t in range(q['root_t']):
                prev=sim.obs(s)['players'][1-s][2];sim.step(*raw['actions'][t])
            p=sim.obs(s)['players'];_,price=inputs(p,s);pos=p[s][2][q['root']['worker']][0]
            x=encode(p,s,pos,prev,q['root_t'],price)
            scores.append(dict(seat=s,root_input=x,head=predict(w,x)[0],type_rate=predict(w,x)[1]))
    def calc(arr):
        return dict(n=len(arr),success=sum(r['bought'] for r in arr),
                    brier={k:np.mean([(r[k]-r['bought'])**2 for r in arr]) for k in ('head','type_rate')})
    assert set(r['seed'] for r in rows).isdisjoint(sr.load(HERE/'payment_result.json')['held_maps'])
    sr.retain(HERE/'transfer_rows.json.gz',rows)
    sr.retain(HERE/'transfer.json',dict(roots=scores,all=calc(rows),threat=calc([r for r in rows if r['threat']])))
if __name__=='__main__':main()
