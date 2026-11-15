"""Direct wiring proof: the value arm's leaf must actually call encode_b
(the value-head path); the const arm must not. Counts calls per arm."""
import sys, os, json, copy
sys.path.insert(0, os.path.abspath("modules"))
import numpy as np
import cnet2_search_vleaf as V
import cnet2_search as B

calls = {"n": 0}
_orig = V.encode_b
def counting(pol, X, teacher):
    calls["n"] += 1
    return _orig(pol, X, teacher)
V.encode_b = counting

W = dict(np.load("smoke_assets/cnet2_m3s2r1L.npz"))
RW = dict(np.load("smoke_assets/cnet2_s64.npz"))
tasks = [json.loads(l) for l in open("smoke/tasks8.jsonl")][:3]

def run(mod, conf, obs):
    calls["n"] = 0
    np.random.seed(20261114)
    ag = mod.make_search_agent(conf=conf, weights=W, rweights=RW)
    a = ag(copy.deepcopy(obs))
    return a, calls["n"]

for t in tasks:
    obs = t["obs"]
    av, nv = run(V, {"LEAF_VALUE": "value"}, obs)
    ac, nc = run(V, {"LEAF_VALUE": "const"}, obs)
    ah, nh = run(B, {}, obs)
    print(f"root {t.get('id', t.get('key','?'))}: value encode_b calls={nv} act={av} | const calls={nc} act={ac} | heur act={ah}")
    assert nv > 0, "value leaf never evaluated!"
    assert nc == 0, "const arm touched the value head!"
print("WIRING_OK")
