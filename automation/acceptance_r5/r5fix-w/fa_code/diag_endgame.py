import json, os, sys, importlib.util
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.join('.', 'env_src', 'carbon_challenge-master'))
from fastsim import FastCarbon
spec = json.load(open("env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json"))
CFG = {}
for k, v in spec["configuration"].items():
    CFG[k] = v.get("default") if isinstance(v, dict) else v
def load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
pv = load("policy_v0_1.py", "pv_eg"); ex = load("example.py", "ex_eg")
sim = FastCarbon(CFG).reset(seed=2001)
a0 = pv.make_agent(); a1 = ex.agent
BASES = {160, 64, 154, 70}  # will print actual below
for t in range(300):
    act0 = a0(sim.obs(0), CFG); act1 = a1(sim.obs(1), CFG)
    sim.step(act0, act1)
    if t >= 254:
        st = sim.get_state()
        if t == 254:
            print("P0 bases:", st["bases"][0], "P1 bases:", st["bases"][1])
        mine = [(w, d["pos"], round(d["carbon"])) for w, d in st["workers"].items()
                if d["player"] == 0 and d["carbon"] > 1]
        print(f"after step {t}: cash={st['cash'][0]:.0f} loaded={mine}")
