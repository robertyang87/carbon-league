import os, sys, time, importlib.util, json
sys.path.insert(0, os.path.abspath('env_src/carbon_challenge-master'))
from zerosum_env import make

def load_agent(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent

def play(path_a, path_b, seed, steps=300):
    a = load_agent(path_a, 'agent_a_%d'%seed)
    b = load_agent(path_b, 'agent_b_%d'%seed)
    env = make("carbon", configuration={"episodeSteps": steps, "randomSeed": seed})
    t0=time.time()
    env.run([a, b])
    dt=time.time()-t0
    j = env.toJSON()
    rewards = [s['reward'] for s in j['steps'][-1]]
    return rewards, j['statuses'], dt

if __name__ == '__main__':
    pa, pb = sys.argv[1], sys.argv[2]
    seeds = [int(x) for x in sys.argv[3].split(',')]
    for s in seeds:
        r, st, dt = play(pa, pb, s)
        print(f"seed={s} A={r[0]:.1f} B={r[1]:.1f} statuses={st} time={dt:.1f}s")
