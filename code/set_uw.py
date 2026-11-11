"""set_uw DS [CAR_W [PLANTER_W [BASE_W [END_W [END_FROM]]]]] -- per-unit weights for a teacher dataset whose cars are NOT to be learned:
uw = PLANTER_W (1) for planter tokens, BASE_W (1) for base tokens, CAR_W (0) for collector tokens; w = 1 for every row,
or END_W for rows with step >= END_FROM (default 250): the teacher's endgame, e.g. Master3s' last 50 steps.
(With BC_SELF_KL in cnet2_train the collectors are held on the initial net instead.)
The recruit mix (base tokens) is learned from a handful of games, the planters' movement is not: a net that has only
the first half recruits fewer planters and loses its tree income. Weight the planters up and check the ledger."""
import os
import sys

import numpy as np

ds = sys.argv[1]; car_w = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
pl_w = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0; base_w = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
kind = np.load(os.path.join(ds, "tok_kind.npy"), mmap_mode="r"); y = np.load(os.path.join(ds, "y.npy"), mmap_mode="r")
N, S = y.shape
uw = np.full((N, S), base_w, np.float32)
uw[:, :10] = np.where(np.asarray(kind[:, :10]) == 0, car_w, pl_w)
end_w = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0; end_from = int(sys.argv[6]) if len(sys.argv) > 6 else 250
w = np.ones(N, np.float32)
if end_w != 1.0:
    w[np.load(os.path.join(ds, "step.npy")) >= end_from] = end_w
np.save(os.path.join(ds, "uw.npy"), uw); np.save(os.path.join(ds, "w.npy"), w)
v = np.asarray(y) >= 0; k = np.asarray(kind[:, :10])
print({"rows": int(N), "planter_tokens": int((v[:, :10] & (k == 1)).sum()), "car_tokens": int((v[:, :10] & (k == 0)).sum()),
       "base_tokens": int(v[:, 10:].sum()), "car_w": car_w, "planter_w": pl_w, "base_w": base_w,
       "end_w": end_w, "end_rows": int((w != 1.0).sum())})
