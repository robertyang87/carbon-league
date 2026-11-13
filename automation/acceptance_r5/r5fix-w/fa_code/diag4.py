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
pv = load("policy_v1.py", "pv_d4"); ex = load("example.py", "ex_d4")
sim = FastCarbon(CFG).reset(seed=42)
a0 = pv.make_agent(); a1 = ex.agent
prev_trees = 0
for t in range(120):
    obs0 = sim.obs(0)
    my_trees = len(obs0["players"][0][3])
    act0 = a0(obs0, CFG); act1 = a1(sim.obs(1), CFG)
    sim.step(act0, act1)
    new_trees = len(sim.get_state()["trees"]) and sum(1 for x in sim.get_state()["trees"].values() if x["player"]==0)
    if new_trees > prev_trees and my_trees >= 9:
        print(f"step {t}: planted while my_tree_count(obs)={my_trees} -> {new_trees}")
    prev_trees = new_trees
