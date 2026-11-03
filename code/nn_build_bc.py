"""nn_build_bc -- behaviour-cloning dataset from public ladder replays.

Input: a shallow clone of JUZI1/carbon-league-replays (index.json,
leaderboard.json, replays/*.json.gz). Only sides played by strategies rated
>= MIN_RATING in that snapshot's leaderboard are used ("teachers").
Per step t the teacher's observation (shared fields from player 0's record,
its own partial `carbon`) is paired with its action recorded at t+1.

Output (npz): X uint8 [n, C, 15, 15] (planes * 127, clipped), Sc float16
[n, N_SCAL], y int8 (0-4 worker moves, 5-7 base NONE/RECC/RECP), kind int8,
gid int32 (game index), step int16, plus games.json metadata.

Usage: python3 nn_build_bc.py REPLAY_DIR OUT.npz [MIN_RATING]
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


def build_game(args):
    path, sides, gi = args
    d = json.load(gzip.open(path))
    st = d["steps"]
    X, Sc, y, kind, step_ = [], [], [], [], []
    for p in sides:
        tr = F.Tracker(p)
        for t in range(len(st) - 1):
            o0 = st[t][0]["observation"]
            obs = {"step": o0["step"], "players": o0["players"], "trees": o0.get("trees"),
                   "carbon": st[t][p]["observation"]["carbon"], "player": p}
            tr.update(obs)
            if st[t][p].get("status") not in (None, "ACTIVE"):
                break
            act = st[t + 1][p].get("action") or {}
            G, scal, _ = tr.global_planes(obs)
            me = obs["players"][p]
            for wid, w in me[2].items():
                a = act.get(wid, "NONE")
                if a not in F.MOVES:
                    continue
                k = 0 if w[2] == "COLLECTOR" else 1
                x, s = F.unit_inputs(G, scal, w[0], k, w[1])
                X.append(x); Sc.append(s); y.append(F.MOVES.index(a)); kind.append(k); step_.append(t)
            for bid, b in me[1].items():
                a = act.get(bid, "NONE")
                if a not in F.BASE_ACTS:
                    continue
                x, s = F.unit_inputs(G, scal, b, 2, 0.0)
                X.append(x); Sc.append(s); y.append(5 + F.BASE_ACTS.index(a)); kind.append(2); step_.append(t)
    if not X:
        return None
    X = np.clip(np.stack(X) * 127.0, 0, 255).astype(np.uint8)
    return (X, np.stack(Sc).astype(np.float16), np.array(y, np.int8), np.array(kind, np.int8),
            np.full(len(y), gi, np.int32), np.array(step_, np.int16))


def main(rdir, out, min_rating=2380.0):
    idx = json.load(open(os.path.join(rdir, "index.json")))
    lb = json.load(open(os.path.join(rdir, "leaderboard.json")))
    strat = lb["strategies"] if isinstance(lb, dict) else lb
    teachers = {r["name"]: r["rating"] for r in strat if (r.get("rating") or 0) >= min_rating}
    tasks, games = [], []
    for g in idx:
        order = g["order"]  # order[side] -> 'a'/'b'
        sides = [s for s in (0, 1) if g["strategies"][order[s]]["name"] in teachers]
        if not sides or g.get("steps", 0) < 250:
            continue
        gi = len(games)
        games.append({"gi": gi, "file": g["file"], "sides": sides,
                      "names": [g["strategies"][order[s]]["name"] for s in (0, 1)],
                      "scores": [g["scores"][order[s]] for s in (0, 1)],
                      "val": zlib.crc32(g["id"].encode()) % 10 == 0})
        tasks.append((os.path.join(rdir, g["file"]), sides, gi))
    parts = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("BC_WORKERS", "3"))) as ex:
        for r in ex.map(build_game, tasks, chunksize=2):
            if r is not None:
                parts.append(r)
    cat = [np.concatenate([p[i] for p in parts]) for i in range(6)]
    np.savez(out, X=cat[0], Sc=cat[1], y=cat[2], kind=cat[3], gid=cat[4], step=cat[5])
    json.dump({"teachers": teachers, "games": games, "min_rating": min_rating},
              open(out.replace(".npz", "_games.json"), "w"), indent=1)
    print(json.dumps({"games": len(games), "teacher_sides": sum(len(g["sides"]) for g in games),
                      "samples": int(len(cat[2])), "val_games": sum(g["val"] for g in games),
                      "label_hist": np.bincount(cat[2], minlength=8).tolist()}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 2380.0)
