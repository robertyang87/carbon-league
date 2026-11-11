"""set_uw DS [CAR_W] -- per-unit weights for a teacher dataset whose cars are NOT to be learned:
uw = 1 for planter and base tokens, CAR_W (default 0) for collector tokens; w = 1 for every row.
(With BC_SELF_KL in cnet2_train the collectors are held on the initial net instead.)"""
import os
import sys

import numpy as np

ds = sys.argv[1]; car_w = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
kind = np.load(os.path.join(ds, "tok_kind.npy"), mmap_mode="r"); y = np.load(os.path.join(ds, "y.npy"), mmap_mode="r")
N, S = y.shape
uw = np.ones((N, S), np.float32)
uw[:, :10] = np.where(np.asarray(kind[:, :10]) == 0, car_w, 1.0)
np.save(os.path.join(ds, "uw.npy"), uw); np.save(os.path.join(ds, "w.npy"), np.ones(N, np.float32))
v = np.asarray(y) >= 0; k = np.asarray(kind[:, :10])
print({"rows": int(N), "planter_tokens": int((v[:, :10] & (k == 1)).sum()), "car_tokens": int((v[:, :10] & (k == 0)).sum()),
       "base_tokens": int(v[:, 10:].sum()), "car_w": car_w})
