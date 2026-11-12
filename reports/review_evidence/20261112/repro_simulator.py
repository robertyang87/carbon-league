"""Read-only simulator review repros. Run with carbon-review-venv/bin/python.

Loads real original package and code; writes synthetic strategies only to a
TemporaryDirectory, and never modifies the reviewed repository.
"""
import copy
import json
import pathlib
import sys
import tempfile
import types

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/env_src/carbon_challenge-master'))
sys.path.insert(0, str(ROOT / 'code'))
from zerosum_env import make, utils
from zerosum_env.envs.carbon import carbon as rules, idgen
from fastsim import FastCarbon, DEFAULT_CONFIG
import eval_v53
import test_fastsim_parity as parity

out = {}
calls = []
def late_only(o, c):
    calls.append(o.step)
    return {next(iter(o.players[o.player][1])): 'RECCOLLECTOR'} if o.step == 299 else {}
e = make('carbon', configuration={'episodeSteps': 300})
e.run([late_only, lambda o, c: {}])
out['original_horizon'] = dict(len_steps=len(e.steps), last_agent_observation=max(calls),
                               final_step=e.steps[-1][0].observation.step,
                               rewards=[x.reward for x in e.steps[-1]])
with tempfile.TemporaryDirectory() as td:
    a = pathlib.Path(td)/'late.py'; b = pathlib.Path(td)/'noop.py'
    a.write_text("def agent(o,c):\n    return {next(iter(o['players'][o['player']][1])):'RECCOLLECTOR'} if o['step']==299 else {}\n")
    b.write_text('def agent(o,c):\n    return {}\n')
    for n in (299, 300):
        eval_v53.NSTEPS = n
        r = eval_v53.play_game(('review', str(a), str(b), 42, 0))
        out[f'local_horizon_{n}'] = {k:r[k] for k in ['my_cash','opp_cash','margin','win','nsteps']}
    a.write_text("def agent(o,c):\n    return {next(iter(o['players'][o['player']][1])):'RECCOLLECTOR'}\n")
    eval_v53.NSTEPS = 2
    r = eval_v53.play_game(('review', str(a), str(b), 42, 0))
    out['same_step_death_ledger'] = r['ledger_pol']
    s = FastCarbon().reset(seed=42)
    bid = s.bases[0][0][0]
    for _ in range(2):
        s.step({bid:'RECCOLLECTOR'}, {})
    out['same_step_death_truth'] = dict(cash=s.cash[0], accepted_recruits=s.worker_counters[0],
                                      survivors=len(s.player_workers[0]), final_step_deaths=len(s.last_collisions), deposits=0)

sim = FastCarbon().reset(full_carbon=[0]*225)
o = sim.full_observation(); o['step']=9; o['players'][0][0]=20; o['players'][1][0]=10
pos=next(iter(o['players'][1][1].values()))
o['players'][1][2]={'player-1-worker-0':[pos,0,'COLLECTOR']}
states=[]
for p in range(2):
    x=copy.deepcopy(o); x['full_carbon']=x['carbon'][:]; x['player']=p
    states.append(utils.structify(dict(observation=x,action={},status='ACTIVE',reward=o['players'][p][0])))
idgen.reset()
r=rules.compute_next_board(states,types.SimpleNamespace(configuration=utils.structify(DEFAULT_CONFIG)))
sim.load_observation(o).step({},{})
out['elimination'] = dict(original=[dict(status=x.status,reward=x.reward) for x in r], local_cash=sim.cash)

parity.check_map_generation(); parity.check_full_game(7); parity.check_full_game(123)
parity.check_scenarios(); parity.check_agent_compat()
out['existing_parity'] = dict(passed=sum(x[2] for x in parity.RESULTS), total=len(parity.RESULTS), failures=parity.MISMATCHES)
print(json.dumps(out, ensure_ascii=False, indent=2))
