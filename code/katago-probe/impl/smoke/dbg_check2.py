import sys, os, json, copy
sys.path.insert(0, os.path.abspath("modules"))
import numpy as np
import cnet2_search_vleaf_dbg as D

W = dict(np.load("smoke_assets/cnet2_m3s2r1L.npz"))
RW = dict(np.load("smoke_assets/cnet2_s64.npz"))
tasks = [json.loads(l) for l in open("smoke/tasks8.jsonl")]
acts = {}
for arm in ("value", "const"):
    for t in tasks[:4]:
        for k in D.DBG: D.DBG[k] = 0
        np.random.seed(20261114)
        # DIAGNOSTIC ONLY: BUDGET/HARD widened so the throttled local VM reaches
        # rollouts; the gate itself runs frozen DEFAULTS.
        ag = D.make_search_agent(conf={"LEAF_VALUE": arm, "BUDGET": 30.0, "HARD": 60.0},
                                 weights=W, rweights=RW)
        a = ag(copy.deepcopy(t["obs"]))
        st = getattr(ag, "last_stats", {}) or {}
        key = (t["step"], t["seat"])
        acts.setdefault(key, {})[arm] = a
        print(f"{arm} step={t['step']} seat={t['seat']} DBG={dict(D.DBG)} "
              f"depth_sum={st.get('depth_sum')} shallow={st.get('shallow',0)} changed_stat={st.get('changed',0)}")
for k, v in acts.items():
    print(k, "value!=const:", v["value"] != v["const"])
