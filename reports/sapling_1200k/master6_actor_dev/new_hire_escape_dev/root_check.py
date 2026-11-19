"""Verify every possible command by replaying the actual joint step in the official engine."""
from pathlib import Path
import sys,copy
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'tree_response_training'))
import select_roots as sr
def check():
    q=sr.load(HERE/'seat0/live_protocol.json');r=q['root'];raw=sr.load(sr.ROOT/q['record']['source']);rows=[]
    for act in ('NONE','UP','RIGHT','DOWN','LEFT'):
        env=sr.live.begin(raw);sim=sr.FastCarbon(raw['game_configuration']).reset(seed=q['seed'])
        for pair in raw['actions'][:r['t']]:env.step(pair);sim.step(*pair)
        pair=copy.deepcopy(raw['actions'][r['t']]);a=pair[0]
        if act=='NONE':a.pop(r['worker'],None)
        else:a[r['worker']]=act
        before=set(sim.workers);env.step(pair);sim.step(*pair)
        assert sr.live.public(env,0)['players']==sim.obs(0)['players']
        rows.append(dict(act=act,removed=sorted(before-set(sim.workers)),collisions=sim.last_collisions,alive=r['worker'] in sim.workers,cash=list(sim.cash)))
    sr.retain(HERE/'root_check.json',rows)
    print([(v['act'],v['removed']) for v in rows])
if __name__=='__main__':check()
