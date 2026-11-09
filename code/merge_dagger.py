"""merge_dagger BASE_DS DAGGER_DIR[,DIR2...] OUT_DS [DAGGER_W] [BASE_FRAC]

env SLOT_W="1:2,3:2" multiplies the base-row sample weight of those teacher slots (DAgger rows untouched).
BASE_FRAC < 1 keeps that fraction of the base TRAIN games (crc of key; all base validation games are kept).

Concatenates a weighted cnet2 dataset (w.npy + uw.npy) with dagger_gen shards into a new dataset dir that
cnet2_train reads as one. DAgger rows get sample weight DAGGER_W (default 1.0) and keep their own unit weights
(relabelled planter slots 3, agreeing planter slots 1, collectors/bases 0), so they only supervise the planter head. 10% of the DAgger games are
validation (crc of the shard name), like the builder does for replay games.
"""
import glob
import json
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cnet2_build import FIELDS  # noqa: E402


SLOT_W = {int(a): float(b) for a, b in (x.split(":") for x in os.environ.get("SLOT_W", "").split(",") if x)}


def main(base, dg, out, dw=1.0, frac=1.0):
    dw = float(dw); frac = float(frac)
    bm = json.load(open(os.path.join(base, "meta.json")))
    B = {k: np.load(os.path.join(base, k + ".npy"), mmap_mode="r") for k in list(FIELDS) + ["w", "uw"]}
    keep_g = [g for g in bm["games"] if g.get("val") or zlib.crc32(g["key"].encode()) % 1000 < frac * 1000]
    keep_ids = np.array(sorted({g["gi"] for g in keep_g}))
    gid = np.asarray(B["gid"])
    rows = np.nonzero(np.isin(gid, keep_ids))[0] if frac < 1 else np.arange(len(gid))
    nb = len(rows)
    shards = sorted(s for d in dg.split(",") for s in glob.glob(os.path.join(d, "shards", "*.npz")))
    lens = [int(np.load(s)["y"].shape[0]) for s in shards]
    nd = sum(lens)
    os.makedirs(out, exist_ok=True)
    fields = dict(FIELDS); fields["w"] = (np.float16, ()); fields["uw"] = (np.float16, B["uw"].shape[1:])
    mm = {k: np.lib.format.open_memmap(os.path.join(out, k + ".npy"), mode="w+", dtype=dt, shape=(nb + nd,) + tuple(sh))
          for k, (dt, sh) in fields.items()}
    for k in mm:                                   # base rows, in chunks
        for a in range(0, nb, 50000):
            b = min(a + 50000, nb)                 # the output is longer than the base: never slice past nb
            r = rows[a:b]
            mm[k][a:b] = B[k][r[0]:r[-1] + 1][r - r[0]]
            if k == "w" and SLOT_W:                # forgetting guard: up-weight the teacher rows of the given style slots
                t = np.asarray(B["teacher"][r[0]:r[-1] + 1][r - r[0]])
                f = np.ones(len(r), np.float32)
                for sl, m in SLOT_W.items():
                    f[t == sl] = m
                mm[k][a:b] = (np.asarray(mm[k][a:b], np.float32) * f).astype(np.float16)
    games = list(bm["games"]); g0 = len(games); o = nb
    for i, (s, n) in enumerate(zip(shards, lens)):
        D = np.load(s)
        for k in FIELDS:
            mm[k][o:o + n] = D[k] if k != "gid" else g0 + i
        mm["uw"][o:o + n] = D["uw"]; mm["w"][o:o + n] = dw
        key = os.path.basename(s)
        games.append({"file": s, "key": key, "names": ["dagger", "dagger"], "scores": [0, 0], "sides": [0],
                      "val": zlib.crc32(key.encode()) % 10 == 0, "src": "dagger", "gi": g0 + i, "n": n})
        o += n
    for k in mm:
        mm[k].flush()
    json.dump({"games": games, "total": nb + nd, "teachers": bm.get("teachers")}, open(os.path.join(out, "meta.json"), "w"))
    print(json.dumps({"base_rows": nb, "dagger_rows": nd, "dagger_games": len(shards), "base_games_kept": len(keep_g), "out": out}))


if __name__ == "__main__":
    main(*sys.argv[1:])
