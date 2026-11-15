import sys, os, json, copy
sys.path.insert(0, os.path.abspath("modules"))
import numpy as np
import cnet2_search_vleaf_dbg as D

W = dict(np.load("smoke_assets/cnet2_m3s2r1L.npz"))
RW = dict(np.load("smoke_assets/cnet2_s64.npz"))
tasks = [json.loads(l) for l in open("smoke/tasks8.jsonl")]
for arm in ("value", "const"):
    for t in tasks[:4]:
        for k in D.DBG: D.DBG[k] = 0
        np.random.seed(20261114)
        ag = D.make_search_agent(conf={"LEAF_VALUE": arm}, weights=W, rweights=RW)
        a = ag(copy.deepcopy(t["obs"]))
        st = getattr(ag, "last_stats", {}) or {}
        rk = {k: st[k] for k in ("searched", "roll", "rolls", "nroll", "depth", "leaves", "nodes", "br", "root_errors") if k in st}
        print(f"{arm} step={t['step']} seat={t['seat']} DBG={dict(D.DBG)} act={a} stats={rk}")
