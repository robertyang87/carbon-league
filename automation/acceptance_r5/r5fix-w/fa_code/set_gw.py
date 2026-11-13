"""set_gw DS FACTOR NAME [NAME ...] -- multiply the row weights (w.npy) of every game whose OTHER player's name contains
one of the NAME substrings. For a teacher dataset built from ladder replays (meta.json keeps names and teacher sides):
weight the teacher's games against the opponents we lose to, so the imitation spends more on how it handles them.
Run after set_uw.py (which rewrites w.npy). Prints the games and rows touched.
  python3 set_gw.py ../data/ds_m3s_w 3 RyanX-AIPowercore Master2 Carago Echo1"""
import json
import os
import sys

import numpy as np

ds, fac, pats = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
meta = json.load(open(os.path.join(ds, "meta.json")))
gid = np.load(os.path.join(ds, "gid.npy")); w = np.load(os.path.join(ds, "w.npy")).astype(np.float32)
hit = {}
for g in meta["games"]:
    others = [n for i, n in enumerate(g["names"]) if i not in g["sides"]]
    for n in others:
        if any(p in n for p in pats):
            hit[g["gi"]] = n
sel = np.isin(gid, list(hit))
w[sel] *= fac
np.save(os.path.join(ds, "w.npy"), w)
cnt = {}
for n in hit.values():
    cnt[n] = cnt.get(n, 0) + 1
print(json.dumps({"games": len(meta["games"]), "games_weighted": len(hit), "rows_weighted": int(sel.sum()), "factor": fac,
                  "by_opponent": cnt, "mean_w": round(float(w.mean()), 3)}, ensure_ascii=False))
