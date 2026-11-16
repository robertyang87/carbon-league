#!/usr/bin/env python3
"""Gate-4 case generation (SYNTHETIC): drive FastCarbon games with stock
SEARCH=False agents (probe-1 modules) and capture observations at steps
296..300 for both seats, recording per-seat worker counts (extinction state).
"""
import json, os, sys
HERE = '/home/hatch/workspace/katago-probe/impl'
sys.path.insert(0, os.path.join(HERE, 'modules'))
import numpy as np
import cnet2_search as base
from fastsim import FastCarbon

A = os.path.join(HERE, 'smoke_assets')
def load(p):
    d = np.load(p, allow_pickle=False)
    return {k: d[k] for k in d.files}
wd = load(os.path.join(A, 'cnet2_m3s2r1L.npz'))
rwd = load(os.path.join(A, 'cnet2_s64.npz'))

def workers(obs, p):
    return len(obs['players'][p][2])

cases = []
for seed in (777, 778, 779, 780):
    sim = FastCarbon(); sim.reset(seed=seed)
    agents = [base.make_search_agent(conf={'SEARCH': False}, weights=wd, rweights=rwd) for _ in range(2)]
    got = {}
    alive300 = None
    for step in range(300):
        obs = [sim.obs(0), sim.obs(1)]
        if step >= 296:
            for seat in (0, 1):
                got[(step, seat)] = {'game_key': f'SYNTH-BND{seed}', 'seat': seat, 'step': step,
                                     'obs': obs[seat],
                                     'workers': [workers(obs[seat], 0), workers(obs[seat], 1)],
                                     'cash': [float(obs[seat]['players'][0][0]), float(obs[seat]['players'][1][0])]}
        a0 = agents[0](obs[0]); a1 = agents[1](obs[1])
        sim.step(a0, a1)
    # step-300 observations (post-final-step state), if the sim still yields obs
    try:
        obs = [sim.obs(0), sim.obs(1)]
        alive300 = [workers(obs[0], 0), workers(obs[0], 1)]
        for seat in (0, 1):
            got[(300, seat)] = {'game_key': f'SYNTH-BND{seed}', 'seat': seat, 'step': 300,
                                'obs': obs[seat], 'workers': alive300,
                                'cash': [float(obs[seat]['players'][0][0]), float(obs[seat]['players'][1][0])]}
    except Exception as e:
        print(f'seed {seed}: step-300 obs unavailable: {e!r}')
    print(f'seed {seed}: captured steps {sorted(set(s for s, _ in got))}, workers@300={alive300}')
    cases.extend(got[k] for k in sorted(got))
    if len(cases) >= 10:
        break

out = '/home/hatch/workspace/katago-probe2/boundary_cases.jsonl'
with open(out, 'w') as f:
    for c in cases:
        f.write(json.dumps(c) + '\n')
print(f'{len(cases)} boundary cases -> {out}')
