"""concat_ds BASE_DS EXTRA_DS OUT_DS [EXTRA_W]

Appends a small cnet2 dataset (e.g. the M2 endgame rows, teacher slot 2) to a big one without rebuilding it.
Both must have the cnet2_build FIELDS plus w.npy and uw.npy. EXTRA rows get their sample weight multiplied by
EXTRA_W (default 1); their game ids are shifted past BASE's games and their meta entries appended, so the
validation split of both datasets is kept as it is."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cnet2_build import FIELDS  # noqa: E402


def main(base, extra, out, ew=1.0):
    ew = float(ew)
    mb, me = (json.load(open(os.path.join(d, "meta.json"))) for d in (base, extra))
    keys = list(FIELDS) + ["w", "uw"]
    B = {k: np.load(os.path.join(base, k + ".npy"), mmap_mode="r") for k in keys}
    E = {k: np.load(os.path.join(extra, k + ".npy"), mmap_mode="r") for k in keys}
    nb, ne = len(B["y"]), len(E["y"])
    os.makedirs(out, exist_ok=True)
    g0 = len(mb["games"])
    for k in keys:
        mm = np.lib.format.open_memmap(os.path.join(out, k + ".npy"), mode="w+", dtype=B[k].dtype, shape=(nb + ne,) + B[k].shape[1:])
        for a in range(0, nb, 50000):
            b = min(a + 50000, nb); mm[a:b] = B[k][a:b]
        ex = np.asarray(E[k])
        if k == "gid":
            ex = ex + g0
        if k == "w":
            ex = (ex.astype(np.float32) * ew).astype(B[k].dtype)
        mm[nb:] = ex.astype(B[k].dtype)
        mm.flush()
    games = list(mb["games"]) + [dict(g, gi=g0 + g["gi"]) for g in me["games"]]
    json.dump({"games": games, "total": nb + ne, "teachers": mb.get("teachers"), "extra": {"from": extra, "w": ew}},
              open(os.path.join(out, "meta.json"), "w"))
    print(json.dumps({"base_rows": nb, "extra_rows": ne, "extra_w": ew, "out": out}))


if __name__ == "__main__":
    main(*sys.argv[1:])
