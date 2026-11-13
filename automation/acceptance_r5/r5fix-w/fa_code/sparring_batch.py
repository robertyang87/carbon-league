import os, sys, time, json, importlib.util
sys.path.insert(0, os.path.abspath('env_src/carbon_challenge-master'))
from zerosum_env import make

def load_agent(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent

def play(pa, pb, seed, tag):
    a = load_agent(pa, f'a_{tag}_{seed}')
    b = load_agent(pb, f'b_{tag}_{seed}')
    env = make("carbon", configuration={"episodeSteps": 300, "randomSeed": seed})
    t0 = time.time()
    env.run([a, b])
    dt = time.time() - t0
    j = env.toJSON()
    r = [s['reward'] for s in j['steps'][-1]]
    return {"tag": tag, "seed": seed, "score_a": r[0], "score_b": r[1], "sec": round(dt, 1)}

if __name__ == '__main__':
    pa, pb, tag, seeds = sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(',')]
    out = open(f'sparring_{tag}.jsonl', 'a')
    for s in seeds:
        res = play(pa, pb, s, tag)
        out.write(json.dumps(res) + '\n'); out.flush()
        print(res, flush=True)
