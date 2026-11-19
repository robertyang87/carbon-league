"""Verify source identity, official terminal, both command tapes and cash-channel differences."""
from collections import Counter
from retest import HERE,sr,live,TREE
from turning_point_ledger import CashEvents

def check(seat):
    out=HERE/f'seat{seat}';case=sr.load(out/'live_protocol.json')
    rawpath=sr.ROOT/case['record']['source']
    assert sr.sha(rawpath)==case['record']['source_sha256']
    raw=sr.load(rawpath);t=case['root_t'];trial=[]
    for arm in ('parent','alternative'):
        rec=sr.load(out/f'live_{arm}.json'); tape=out/rec['action_tape']
        clock=sr.load(out/f'clock_{arm}.json.gz')
        assert sr.sha(tape)==rec['action_sha256']==clock['action_sha256']
        assert rec['pack_sha256']==case['pack_sha256'] and rec['source_sha256']==live.MASTER_SHA
        assert rec['shared_init_sha256']==sr.sha(out/'live_shared_init.json')
        env=live.begin(raw)
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=case['seed'])
        for pair in raw['actions'][:t]:env.step(pair);sim.step(*pair)
        start=list(sim.cash);sim.cash=CashEvents(start);money=[Counter(),Counter()]
        tail=sr.load(tape)
        worker=case['root']['worker'];target=sim.workers[worker].pos
        focal=dict(root_orders=[],next_pos=None,site=target,root_expenses=[],tree=None,income_by_seat=[0.,0.])
        for turn,pair in enumerate(tail):
            assert not env.done
            env.step(pair);sim.cash.events.clear();sim.step(*pair)
            if turn==0:
                focal["next_pos"]=sim.workers[worker].pos if worker in sim.workers else None
                focal["root_orders"]=pair
            for cat,s,delta,*extra in sim.cash.events:
                money[s][cat]+=delta
                if turn==0 and cat in ("plant","seize") and s==seat and extra[0]==worker:
                    focal["root_expenses"].append(dict(kind=cat,change=delta))
                    focal["tree"]=next(tid for tid,tr in sim.trees.items() if tr.pos==extra[1])
            if focal["tree"] in sim.trees:
                tr=sim.trees[focal["tree"]]
                focal["income_by_seat"][tr.player]+=sim.trees_dict[tr.id][1]
        assert env.done and [p.reward for p in env.states]==rec['score']==list(sim.cash)
        assert all(abs(start[s]+sum(money[s].values())-sim.cash[s])<1e-5 for s in (0,1))
        assert len(clock['calls'])==len(tail)-1==rec['search_calls']
        assert tail[0][1-seat]==raw['actions'][t][1-seat]
        trial.append(dict(arm=arm,margin=rec['margin'],scores=rec['score'],money=[dict(v) for v in money],
                          focal=focal,root_command=tail[0][seat],calls=len(clock['calls']),cutoffs=clock['summary']['native_aborted']))
    old,new=trial
    changes={k for k in old['root_command'].keys()|new['root_command'].keys() if old['root_command'].get(k)!=new['root_command'].get(k)}
    assert changes=={case['root']['worker']}
    keys=set().union(*old['money'],*new['money'])
    diffs={k:sum((1 if s==seat else -1)*(new['money'][s].get(k,0)-old['money'][s].get(k,0)) for s in (0,1)) for k in keys}
    gain=new['margin']-old['margin'];assert abs(sum(diffs.values())-gain)<1e-5
    return dict(seed=case['seed'],seat=seat,root=case['root'],historical=case['observed_original'],
                fresh=gain,arms=trial,net_channels=diffs)

if __name__=='__main__':
    res=[check(s) for s in (0,1)]
    sr.retain(HERE/'retest_result.json',res)
    for r in res:
        print(r['seat'],r['historical'],r['fresh'],[(a['margin'],a['calls'],a['cutoffs']) for a in r['arms']])
