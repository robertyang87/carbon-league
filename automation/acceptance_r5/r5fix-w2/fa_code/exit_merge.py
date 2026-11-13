"""exit_merge -- merge data/exit1/shards/*.npz (exit_gen) into a cnet2_build-style directory (one .npy per field,
memmap-able, meta.json with games + val flags) that cnet2_exit_train reads.  Extra fields: searched, changed,
pi, pi_slot, y_net, q.  Validation games: seed % 10 == 0.
  python3 exit_merge.py OUT_DIR [SHARD_DIR ...]"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cnet2_build import FIELDS  # noqa: E402
from exit_gen import EXTRA  # noqa: E402


EXCL = [x for x in os.environ.get("EXIT_EXCLUDE", "cnet2_v13,opp_1cz_v5,b160").split(",") if x]   # held-out test opponents


def main(out, dirs):
    os.makedirs(out, exist_ok=True)
    files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "*.npz")))
    games, ns = [], []
    for fn in files:
        try:
            with np.load(fn) as D:
                n = int(D["y"].shape[0])
        except Exception:
            continue
        b = os.path.basename(fn)[:-4]
        seed = int(b.split("_")[0])
        if any(x in b for x in EXCL):
            continue
        games.append({"shard": fn, "key": b, "seed": seed, "opp": "_".join(b.split("_")[1:-1]), "val": seed % 10 == 0, "n": n})
        ns.append(n)
    total = int(sum(ns))
    fields = dict(FIELDS); fields.update(EXTRA)
    mm = {k: np.lib.format.open_memmap(os.path.join(out, k + ".npy"), mode="w+", dtype=dt, shape=(total,) + sh)
          for k, (dt, sh) in fields.items()}
    o = 0
    for gi, g in enumerate(games):
        with np.load(g["shard"]) as D:
            n = g["n"]
            for k in fields:
                if k == "gid":
                    mm[k][o:o + n] = gi
                elif k in D.files:
                    mm[k][o:o + n] = D[k]
                else:
                    mm[k][o:o + n] = np.nan if k == "q" else 0
        g["gi"] = gi
        o += n
    for k in mm:
        mm[k].flush()
    ch = np.asarray(mm["changed"]); se = np.asarray(mm["searched"])
    json.dump({"games": games, "total": total}, open(os.path.join(out, "meta.json"), "w"))
    print(json.dumps({"games": len(games), "samples": total, "val_games": sum(g["val"] for g in games),
                      "searched": int(se.sum()), "changed": int(ch.sum())}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or [os.path.join(HERE, "..", "data", "exit1", d) for d in ("shards", "shards_mfxn", "shards_mfh")])
