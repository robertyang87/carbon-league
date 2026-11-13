import json, os, sys, importlib.util
sys.path.insert(0, os.path.abspath('.'))
from fastsim import FastCarbon
spec = json.load(open("env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json"))
CFG = {}
for k, v in spec["configuration"].items():
    CFG[k] = v.get("default") if isinstance(v, dict) else v
def load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
pv = load("policy_v1.py", "pv_d2"); ex = load("example.py", "ex_d2")
sim = FastCarbon(CFG).reset(seed=42)
a0 = pv.make_agent(); a1 = ex.agent
for t in range(160):
    pre = {w: dict(d) for w, d in sim.get_state()["workers"].items() if d["player"] == 0}
    act0 = a0(sim.obs(0), CFG); act1 = a1(sim.obs(1), CFG)
    sim.step(act0, act1)
    post_state = sim.get_state()
    post = post_state["workers"]
    for w, d in pre.items():
        if w not in post:
            at = [(ww, dd["type"], dd["carbon"]) for ww, dd in post.items() if dd["pos"] == d["pos"]]
            print(f"step {t}: DIED {w} {d['type']} cargo={d['carbon']:.0f} at pos={d['pos']} action={act0.get(w)} survivors_there={at} cash={post_state['cash'][0]:.0f}")
