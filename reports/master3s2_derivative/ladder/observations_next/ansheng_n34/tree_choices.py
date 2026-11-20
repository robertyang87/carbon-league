"""Pair a costly plant against every physical first move and two same-turn plant controls."""
import copy
import gzip
import json
from pathlib import Path
import random
import runpy
import sys

HERE = Path(__file__).resolve().parent
LADDER = HERE.parents[1]
sys.path.insert(0, str(LADDER.parents[1] / 'sapling_1200k/all_top5'))
from economy_extract import FastCarbon

GAME = 'eec452f455f16f82bb432ce057daa25c'
CASES = {137: ('player-1-worker-22','player-1-worker-24'), 142: ('player-1-worker-24',)}
MOVES = ('NONE','UP','RIGHT','DOWN','LEFT')


def physical(frames):
    obs=frames[0]['observation']
    return obs['step'],obs['players'],obs['trees'],obs['full_carbon']


def run():
    engine=runpy.run_path(str(LADDER.parent / 'freight_response/run.py'))
    replay=json.loads(gzip.decompress((LADDER/'replays'/f'{GAME}-1.json.gz').read_bytes()))
    frames=replay['steps']
    cfg=replay['configuration']
    random.seed(cfg['randomSeed'])
    engine['np'].random.seed(cfg['randomSeed'])
    env=engine['make']('carbon',configuration=cfg)
    terrain=frames[0][0]['observation']['full_carbon']
    env.states[0].observation.full_carbon=copy.deepcopy(terrain)
    sim=FastCarbon(cfg).reset(full_carbon=terrain)
    results=[]
    for t in range(max(CASES)+1):
        assert physical(env.states)==physical(frames[t])
        actions=[dict(x['action'] or {}) for x in frames[t+1]]
        if t in CASES:
            env.steps[:-1]=[None]*t
            env.logs=[]
            root=engine['official_state'].take(env,[0,0])
            for uid in CASES[t]:
                assert actions[1].get(uid,'NONE')=='NONE'
                records=[]
                for first in MOVES:
                    engine['official_state'].restore(env,root)
                    branch=copy.deepcopy(sim)
                    trace=[]
                    for j in range(8):
                        cmd=[dict(x['action'] or {}) for x in frames[t+j+1]]
                        if j==0 and first!='NONE':
                            cmd[1][uid]=first
                        env.step(cmd,[])
                        branch.step(*cmd)
                        obs=env.states[0].observation
                        assert obs.players==branch.obs(0)['players'] and list(branch.carbon)==obs.full_carbon
                        if first=='NONE':
                            assert physical(env.states)==physical(frames[t+j+1])
                        trace.append(dict(t=t+j+1,margin=round(branch.cash[1]-branch.cash[0],2),cash=list(branch.cash),
                                          worker=obs.players[1][2].get(uid), collisions=branch.last_collisions))
                    records.append(dict(first=first,trace=trace))
                results.append(dict(t=t,worker=uid,routes=records))
            engine['official_state'].restore(env,root)
        sim.step(*actions)
        env.step(actions,[])
    return dict(game=GAME,seat=1,method="changed one focal first order only; all other simultaneous and later orders historically fixed",cases=results)


if __name__=='__main__':
    report=run()
    (HERE/'tree_choices.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    for case in report['cases']:
        print(case['t'],case['worker'],[(b['first'],b['trace'][0]['margin'],b['trace'][-1]['margin']) for b in case['routes']])
