import json, os, sys, importlib.util
sys.path.insert(0, os.path.abspath('.'))
from fastsim import FastCarbon
import policy_v1 as PVMOD
spec = json.load(open("env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json"))
CFG = {}
for k, v in spec["configuration"].items():
    CFG[k] = v.get("default") if isinstance(v, dict) else v
def load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ex = load("example.py", "ex_d3")
sim = FastCarbon(CFG).reset(seed=42)
a0 = PVMOD.make_agent(); a1 = ex.agent
OFF = {"UP":0,"RIGHT":1,"DOWN":2,"LEFT":3}
for t in range(160):
    pre = {w: dict(d) for w, d in sim.get_state()["workers"].items() if d["player"] == 0}
    act0 = a0(sim.obs(0), CFG); act1 = a1(sim.obs(1), CFG)
    sim.step(act0, act1)
    post_state = sim.get_state(); post = post_state["workers"]
    for w, d in pre.items():
        if w not in post:
            a = act0.get(w)
            dest = d["pos"] if a is None else int(PVMOD.NBR4[d["pos"], OFF[a]])
            at = [(ww, dd["player"], dd["type"], round(dd["carbon"])) for ww, dd in post.items() if dd["pos"] == dest]
            print(f"step {t}: DIED {w} {d['type']} cargo={d['carbon']:.0f} dest={dest} (base={dest in (64,160,70,154)}) there_now={at}")
