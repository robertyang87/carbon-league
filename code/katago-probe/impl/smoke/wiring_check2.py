import sys, os, json, copy
sys.path.insert(0, os.path.abspath("modules"))
import numpy as np
import cnet2_search_vleaf as V

cnt = {"leaf": 0, "encb": 0}
_ol, _oe = V.leaf, V.encode_b
def leaf_c(*a, **k):
    cnt["leaf"] += 1
    return _ol(*a, **k)
def encb_c(*a, **k):
    cnt["encb"] += 1
    return _oe(*a, **k)
V.leaf, V.encode_b = leaf_c, encb_c

W = dict(np.load("smoke_assets/cnet2_m3s2r1L.npz"))
RW = dict(np.load("smoke_assets/cnet2_s64.npz"))
tasks = [json.loads(l) for l in open("smoke/tasks8.jsonl")][:3]
for t in tasks:
    cnt.update(leaf=0, encb=0)
    np.random.seed(20261114)
    ag = V.make_search_agent(conf={"LEAF_VALUE": "value"}, weights=W, rweights=RW)
    a = ag(copy.deepcopy(t["obs"]))
    st = getattr(ag, "last_stats", {}) or {}
    keys = {k: st[k] for k in sorted(st) if isinstance(st[k], (int, float)) and k not in ("q",)}
    print(f"step={t['step']} seat={t['seat']} leaf_calls={cnt['leaf']} encode_b_calls={cnt['encb']} act={a}")
    print("   stats:", keys)
