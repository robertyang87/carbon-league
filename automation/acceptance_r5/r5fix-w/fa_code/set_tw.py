"""set_tw DS ROW=W [ROW=W ...] -- multiply the row weights (w.npy) of every sample whose teacher (style row) is ROW.
For multi-teacher datasets (build_top.py): keep one teacher from dominating the shared trunk, or weight the row we
play. Run after set_uw.py (which rewrites w.npy). Prints rows and weight share per teacher before and after.
  python3 set_tw.py ../data/ds_top 2=3 6=0.3"""
import json
import os
import sys

import numpy as np

ds = sys.argv[1]
mul = {int(a.split("=")[0]): float(a.split("=")[1]) for a in sys.argv[2:]}
t = np.load(os.path.join(ds, "teacher.npy")).astype(np.int64)
w = np.load(os.path.join(ds, "w.npy")).astype(np.float32)
names = json.load(open(os.path.join(ds, "meta.json"))).get("teachers", [])
before = {int(k): float(w[t == k].sum()) for k in np.unique(t)}
for k, m in mul.items():
    w[t == k] *= m
np.save(os.path.join(ds, "w.npy"), w)
tot0, tot1 = sum(before.values()), float(w.sum())
print(json.dumps({"teachers": {str(int(k)): {"name": names[int(k)] if int(k) < len(names) else "?", "rows": int((t == k).sum()),
                                             "share_before": round(before[int(k)] / tot0, 3),
                                             "share_after": round(float(w[t == k].sum()) / tot1, 3)} for k in np.unique(t)}},
                 ensure_ascii=False))
