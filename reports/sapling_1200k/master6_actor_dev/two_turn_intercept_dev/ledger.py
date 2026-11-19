"""Verify official scores, realized freight collisions, focal payments and both teams' cash flows."""
from collections import Counter
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'tree_response_training'))
import select_roots as sr
from turning_point_ledger import CashEvents

def check(out,seat,arm):
    q=sr.load(out/'live_protocol.json');r=q['root']
    raw=sr.load(sr.ROOT/q['record']['source'])
    rec=sr.load(out/f'live_{arm}.json')
    tape=out/rec['action_tape'];wal=sr.load(out/f'clock_{arm}.json.gz')
    assert sr.sha(tape)==rec['action_sha256']==wal['action_sha256']
    assert rec['source_sha256']==sr.live.MASTER_SHA and rec['pack_sha256']==q['pack_sha256']
    env=sr.live.begin(raw)
    sim=sr.FastCarbon(raw['game_configuration']).reset(seed=q['seed'])
    for pair in raw['actions'][:r['t']]:sim.step(*pair);env.step(pair)
    sim.cash=CashEvents(sim.cash);initial=list(sim.cash)
    flows=[Counter(),Counter()];windows={};focal=[];coll=[];banks=[];actions=sr.load(tape)
    assert actions[0][1-seat]==raw['actions'][r['t']][1-seat]
    for i,pair in enumerate(actions,1):
        old={k:[sim.workers[k].pos,sim.workers[k].carbon] if k in sim.workers else None for k in (r['worker'],r['target'])}
        sim.cash.events.clear();sim.step(*pair);env.step(pair)
        for kind,side,amount,*extra in sim.cash.events:
            flows[side][kind]+=amount
            if kind=='own_delivery' and extra[0] in old:banks.append([i,extra[0],amount])
        for item in sim.last_collisions:
            if item[0] in old:coll.append([i,*item])
        if i<=5:
            focal.append(dict(i=i,before=old,orders=[pair[seat].get(r['worker']),pair[1-seat].get(r['target'])],
                              after={k:[sim.workers[k].pos,sim.workers[k].carbon] if k in sim.workers else None for k in old}))
        if i in {1,2,5,10,30,len(actions)}:
            assert all(abs(initial[s]+sum(flows[s].values())-sim.cash[s])<1e-5 for s in (0,1))
            windows[str(i)]=dict(cash=list(sim.cash),flows=[dict(x) for x in flows])
    assert env.done and [x.reward for x in env.states]==rec['score']==list(sim.cash)
    assert len(wal['calls'])==len(actions)-1==rec['search_calls']
    return dict(seat=seat,seed=q['seed'],arm=arm,root=r,margin=rec['margin'],score=rec['score'],windows=windows,
                collisions=coll,deposits=banks,first_five=focal,calls=len(wal['calls']),cutoffs=wal['summary']['native_aborted'])

def summarize(directory):
    rows=[check(directory/f'seat{s}',s,a) for s in (0,1) for a in ('parent','approach')]
    sr.retain(directory/'ledger.json',rows)
    for r in rows:
        print(r['seat'],r['arm'],r['margin'],r['collisions'],r['deposits'])
if __name__=='__main__':summarize(HERE)
