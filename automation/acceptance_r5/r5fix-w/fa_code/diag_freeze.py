import json, os, sys, importlib.util
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.join('.', 'env_src', 'carbon_challenge-master'))
from fastsim import FastCarbon
spec = json.load(open("env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json"))
CFG = {}
for k, v in spec["configuration"].items():
    CFG[k] = v.get("default") if isinstance(v, dict) else v
def load(path, name, kind):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m.make_agent() if kind == "policy" else m.agent
a0 = load("policy_v0_1.py", "pv_fz", "policy")
a1 = load("reference_juzi.py", "juzi_fz", "opp")
sim = FastCarbon(CFG).reset(seed=1001)
bases0 = {pos for _, pos in sim.bases[0]}
runs = {}   # wid -> [cur_run, max_run, frozen_steps_total]
for t in range(300):
    pre = {w: (d["pos"], d["carbon"]) for w, d in sim.get_state()["workers"].items()
           if d["player"] == 0 and d["type"] == "COLLECTOR"}
    sim.step(a0(sim.obs(0), CFG), a1(sim.obs(1), CFG))
    post = sim.get_state()["workers"]
    for wid, (pos, cargo) in pre.items():
        if wid not in post:
            continue
        p2, c2 = post[wid]["pos"], post[wid]["carbon"]
        r = runs.setdefault(wid, [0, 0, 0])
        if p2 == pos and cargo >= 100 and pos not in bases0:
            r[0] += 1
            r[1] = max(r[1], r[0])
            r[2] += 1
        else:
            r[0] = 0
print("wid: [_, longest stationary-loaded run, total stationary-loaded steps]")
tot = 0
for wid, r in sorted(runs.items()):
    if r[1] >= 5:
        print(wid, r)
    tot += r[2]
print("total stationary-loaded collector-steps:", tot)
st = sim.get_state()
print("final cash:", st["cash"])
