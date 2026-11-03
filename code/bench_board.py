import os, sys, time, random
sys.path.insert(0, os.path.abspath('env_src/carbon_challenge-master'))
from zerosum_env import make

# measure env.run with agents that actually play: use reference-lite = random mover with recruiting via raw obs
def fast_random(obs, config):
    me = obs['players'][obs['player']]
    cash, centers, workers = me[0], me[1], me[2]
    actions = {}
    for wid in workers:
        actions[wid] = random.choice(['UP','RIGHT','DOWN','LEFT', None])
        actions[wid] = random.choice(['UP','RIGHT','DOWN','LEFT'])
    for cid in centers:
        if cash >= 30:
            actions[cid] = random.choice(['RECCOLLECTOR','RECPLANTER'])
            cash -= 30
    return actions

env = make("carbon", configuration={"episodeSteps": 300, "randomSeed": 3})
t0=time.time(); env.run([fast_random, fast_random]); dt=time.time()-t0
j=env.toJSON(); print("fast_random game:", [s['reward'] for s in j['steps'][-1]], f"{dt:.1f}s")
