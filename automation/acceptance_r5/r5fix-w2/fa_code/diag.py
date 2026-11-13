import json, os, sys, time, importlib.util
sys.path.insert(0, os.path.abspath('.'))
from fastsim import FastCarbon
spec = json.load(open("env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json"))
CFG = {}
for k, v in spec["configuration"].items():
    CFG[k] = v.get("default") if isinstance(v, dict) else v
def load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
pv = load("policy_v1.py", "pv_diag"); ex = load("example.py", "ex_diag")
sim = FastCarbon(CFG).reset(seed=42)
a0 = pv.make_agent(); a1 = ex.agent
prev_workers = set()
deaths = 0
for t in range(300):
    act0 = a0(sim.obs(0), CFG); act1 = a1(sim.obs(1), CFG)
    sim.step(act0, act1)
    st = sim.get_state()
    cur = {w for w, d in st["workers"].items() if d["player"] == 0}
    deaths += len(prev_workers - cur)
    prev_workers = cur
    if t % 50 == 49 or t == 299:
        mine = [d for d in st["workers"].values() if d["player"] == 0]
        print(f"step {t+1}: cash={st['cash'][0]:.0f} workers={len(mine)} "
              f"(C={sum(1 for d in mine if d['type']=='COLLECTOR')},P={sum(1 for d in mine if d['type']=='PLANTER')}) "
              f"cargo={sum(d['carbon'] for d in mine):.0f} trees={sum(1 for x in st['trees'].values() if x['player']==0)} cum_deaths={deaths}")
import cProfile, pstats, io
sim2 = FastCarbon(CFG).reset(seed=7)
a2 = pv.make_agent()
pr = cProfile.Profile(); pr.enable()
for t in range(60):
    sim2.step(a2(sim2.obs(0), CFG), {})
pr.disable()
s = io.StringIO(); ps = pstats.Stats(pr, stream=s).sort_stats("cumulative"); ps.print_stats(12)
print("\n".join(s.getvalue().splitlines()[4:22]))
