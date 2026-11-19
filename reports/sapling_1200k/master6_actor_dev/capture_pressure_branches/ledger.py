"""Replay both action tapes, measure enemy response near the target, and reconcile all cash."""
from collections import Counter
from compare import HERE,sr
from turning_point_ledger import CashEvents

def dist(a,b):
    y,x=abs(a//15-b//15),abs(a%15-b%15)
    return min(y,15-y)+min(x,15-x)

def trace(seat,arm):
    out=HERE/f'seat{seat}'
    case=sr.load(out/'live_protocol.json')
    raw=sr.load(sr.ROOT/case['record']['source'])
    result=sr.load(out/f'live_{arm}.json')
    tape=out/result['action_tape']
    assert sr.sha(tape)==result['action_sha256']==sr.load(out/f'clock_{arm}.json.gz')['action_sha256']
    orders=sr.load(tape)
    sim=sr.FastCarbon(raw['game_configuration']).reset(seed=case['seed'])
    prefix=[]
    for t,pair in enumerate(raw['actions'][:case['root_t']]):
        if case['root_t']-t in (10,5,3):
            obs=sim.obs(seat)
            prefix.append(dict(t=t,players=obs['players'],trees=obs['trees'],visible_carbon=obs['carbon'],orders=pair))
        sim.step(*pair)
    tid=case['root']['tree'];wid=case['root']['worker'];cell=sim.trees[tid].pos
    root_context=dict(site=cell,age=sim.trees[tid].age,last_tree_payment=sim.trees_dict[tid][1],
                      visible=sim.obs(seat),prefix=prefix)
    start=list(sim.cash);sim.cash=CashEvents(start)
    money=[Counter(),Counter()];detail=[];snap={}
    nearby={uid for uid,w in sim.workers.items() if w.player!=seat and dist(w.pos,cell)<=2}
    deposits=[];target_income=[0.,0.];target_turns=[0,0]
    for offset,pair in enumerate(orders,1):
        positions={uid:sim.workers[uid].pos for uid in nearby|{wid} if uid in sim.workers}
        sim.cash.events.clear();sim.step(*pair)
        for kind,s,delta,*extra in sim.cash.events:
            money[s][kind]+=delta
            if kind=='own_delivery' and extra[0] in nearby:
                deposits.append(dict(offset=offset,worker=extra[0],amount=delta))
        target=sim.trees.get(tid)
        if target:
            target_turns[target.player]+=1
            target_income[target.player]+=sim.trees_dict[tid][1]
        if offset<=10:
            detail.append(dict(offset=offset,target_owner=target.player if target else None,
                actors=[dict(worker=uid,pos=pos,order=pair[seat if uid==wid else 1-seat].get(uid),
                    after=sim.workers[uid].pos if uid in sim.workers else None) for uid,pos in sorted(positions.items())]))
        if offset in {1,3,5,10,30,len(orders)}:
            for s in (0,1):assert abs(start[s]+sum(money[s].values())-sim.cash[s])<1e-5
            snap[str(offset)]=dict(accounts=[dict(m) for m in money],cash=list(sim.cash),
                                   target_turns=target_turns[:],target_income=target_income[:])
    assert list(sim.cash)==result['score']
    return dict(seat=seat,arm=arm,root_context=root_context,root=case['root'],margin=result['margin'],score=result['score'],
                snapshots=snap,nearby_opponents=sorted(nearby),their_deposits=deposits,first_ten=detail)

if __name__=='__main__':
    sr.retain(HERE/'ledger.json',[trace(s,a) for s in (0,1) for a in ('parent','approach')])
