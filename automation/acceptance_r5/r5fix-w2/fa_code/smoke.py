import json, os, sys, time, importlib.util
sys.path.insert(0, os.path.abspath('.'))
from fastsim import FastCarbon

spec = json.load(open("env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json"))
CFG = {}
for k, v in spec["configuration"].items():
    CFG[k] = v.get("default") if isinstance(v, dict) else v

def load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m

pv = load("policy_v1.py", "pv_smoke")
ex = load("example.py", "ex_smoke")
sim = FastCarbon(CFG).reset(seed=42)
a0 = pv.make_agent(); a1 = ex.agent
t_mine = 0.0
t0 = time.time()
for t in range(300):
    o0, o1 = sim.obs(0), sim.obs(1)
    ts = time.time(); act0 = a0(o0, CFG); t_mine += time.time() - ts
    act1 = a1(o1, CFG)
    sim.step(act0, act1)
print("vs example: cash", sim.cash, "wall", round(time.time()-t0,1), "s; my decision avg ms/step", round(t_mine/300*1000,2), "max workers", )
st = sim.get_state()
print("my stranded cargo:", sum(w["carbon"] for w in st["workers"].values() if w["player"]==0))
print("my workers:", sum(1 for w in st["workers"].values() if w["player"]==0), "trees:", sum(1 for t in st["trees"].values() if t["player"]==0))
