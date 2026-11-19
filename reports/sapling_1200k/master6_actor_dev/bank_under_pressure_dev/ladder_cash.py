"""Verify every cash transition, then attribute the 3/5/10 turn largest cash swing."""
import json,gzip
from collections import Counter
from intake import HERE,ladder
from audit import sr
from turning_point_ledger import CashEvents

def examine(g):
    env,_,_=ladder.fetch.extract((ladder.fetch.CACHE/'replays'/(g['id']+'.json.gz')).read_bytes())
    assert ladder.fetch.sha(ladder.fetch.canonical(env))==g['canonical_sha256']
    steps=env['steps'];cfg=env['configuration'];o=steps[0][0]['observation']
    sim=sr.FastCarbon(cfg).reset(seed=cfg['randomSeed'],full_carbon=o['full_carbon'],cash=o['players'][0][0])
    sim.cash=CashEvents(sim.cash); events=[]; totals=[Counter(),Counter()];frames=[]
    for t,(before,after) in enumerate(zip(steps,steps[1:])):
        cmds=[v.get('action') or {} for v in after]
        sim.cash.events.clear();sim.step(*cmds)
        expected=[p[0] for p in after[0]['observation']['players']]
        if list(sim.cash)!=expected:
            raise ValueError((g['id'],t,list(sim.cash),expected))
        events.append(list(sim.cash.events))
        for k,s,v,*_ in sim.cash.events:totals[s][k]+=v
        if g['end']-10<=t<g['end']:
            frames.append(dict(t=t,players=before[0]['observation']['players'],commands=cmds,flows=events[-1]))
    windows={}
    for n in (3,5,10):
        flows=[Counter(),Counter()]
        for batch in events[g['end']-n:g['end']]:
            for k,s,v,*_ in batch:flows[s][k]+=v
        windows[str(n)]=[dict(x) for x in flows]
    return dict(id=g['id'],steps=len(events),totals=totals,windows=windows,frames=frames)
if __name__=='__main__':
    games=[g for name in ('master5s_delta','additional_indices') for g in json.loads((HERE/(name+'.json')).read_bytes())['games']]
    sr.retain(HERE/'ladder_cash.json.gz',[examine(g) for g in games])
