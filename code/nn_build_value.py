"""nn_build_value -- (global planes, scalars) -> final margin dataset for a state-value network.

For every replay in the given snapshot-format dirs and both players, replay that player's partial
observation through nnfeat.Tracker (exactly as at inference) and record every STRIDE-th step:
  G   float16 [n, C_PLANES, 15, 15]  (non-egocentric global planes)
  Sc  float16 [n, N_SCAL]
  y   float32 [n]  final margin from this player's view / 5000
  step int16, gid int32, val bool (game-level split by crc32(id) % 10 == 0)

Usage: python3 nn_build_value.py OUT.npz DIR [DIR ...]
"""
import gzip
import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nnfeat as F  # noqa: E402

STRIDE = int(os.environ.get("VAL_STRIDE", "3"))


def build(args):
    path, gid, gkey = args
    try:
        d = json.load(gzip.open(path))
    except Exception:
        return None
    st = d["steps"]
    if len(st) < 250:
        return None
    rewards = d.get("rewards") or [None, None]
    fin = [st[-1][0]["observation"]["players"][p][0] for p in (0, 1)]
    G_, S_, y_, t_ = [], [], [], []
    for p in (0, 1):
        margin = (fin[p] - fin[1 - p]) / 5000.0
        tr = F.Tracker(p)
        for t in range(len(st) - 1):
            o0 = st[t][0]["observation"]
            obs = {"step": o0["step"], "players": o0["players"], "trees": o0.get("trees"),
                   "carbon": st[t][p]["observation"]["carbon"], "player": p}
            tr.update(obs)
            if t % STRIDE:
                continue
            G, scal, _ = tr.global_planes(obs)
            G_.append(G.reshape(F.C_PLANES, 15, 15)); S_.append(scal); y_.append(margin); t_.append(t)
    val = zlib.crc32(gkey.encode()) % 10 == 0
    return (np.stack(G_).astype(np.float16), np.stack(S_).astype(np.float16),
            np.array(y_, np.float32), np.array(t_, np.int16), np.full(len(y_), gid, np.int32),
            np.full(len(y_), val, bool))


def main(out, dirs):
    tasks, seen = [], set()
    for d in dirs:
        for g in json.load(open(os.path.join(d, "index.json"))):
            k = (g["id"], g["game"])
            if k in seen or g.get("steps", 0) < 250:
                continue
            seen.add(k)
            f = g["file"] if os.path.isabs(g["file"]) else os.path.join(d, g["file"])
            tasks.append((f, len(tasks), g["id"]))
    parts = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("VAL_WORKERS", "4"))) as ex:
        for r in ex.map(build, tasks, chunksize=4):
            if r is not None:
                parts.append(r)
    G, S, y, t, gid, val = (np.concatenate([p[i] for p in parts]) for i in range(6))
    np.savez(out, G=G, Sc=S, y=y, step=t, gid=gid, val=val)
    print(json.dumps({"games": len(parts), "samples": int(len(y)), "val_frac": float(val.mean()),
                      "y_mean": float(y.mean()), "y_std": float(y.std())}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
