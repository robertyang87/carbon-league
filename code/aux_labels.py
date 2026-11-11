"""aux_labels DS -- auxiliary prediction targets for a cnet2_build dataset, read from the replays (nothing estimated
except the tree proxy). Stage A of reports/plan_katago.md: dense, low-noise targets next to the policy labels.

Written into DS (N rows, aligned with y.npy):
  aux_die.npy    [N, 10, 3] uint8    own unit in slot s is destroyed within 10 / 25 / 50 steps
  aux_die_m.npy  [N, 10, 3] bool     valid: slot occupied and the horizon ends inside the game
  aux_dep.npy    [N, 10, 2] float16  carbon this collector deposits within 25 / 50 steps, / 100
  aux_dep_m.npy  [N, 10, 2] bool     valid: slot holds a collector and the horizon ends inside the game
  aux_dw.npy     [N, 3]     float16  change of (our worth - their worth) over 10 / 25 / 50 steps, / 1000, clipped to +-3
  aux_dw_m.npy   [N, 3]     bool     valid: the horizon ends inside the game
worth = cash + cargo on board + remaining value of standing trees
        (4-neighbour carbon * (1 - 0.9625^min(50 - age, steps left, 30)) * 0.8; needs full_carbon in the replay,
        otherwise the tree term is left out for that game and counted in the summary).

Rows are matched to the replay by (game, side, step) from meta.json / gid.npy / step.npy; a slot is matched to its unit
by position (tok_pos is the canonical cell: mirrored for seat 1).
  python3 aux_labels.py DS [WORKERS]"""
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
KD = (10, 25, 50); KP = (25, 50)


def _worth(o, full, p, left):
    P = o["players"][p]
    v = float(P[0]) + sum(float(w[1]) for w in P[2].values())
    if full is not None:
        for pos, age in P[3].values():
            r, c = divmod(int(pos), 15)
            nb = sum(full[((r + dr) % 15) * 15 + (c + dc) % 15] for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
            v += nb * (1.0 - 0.9625 ** max(0, min(50 - int(age), left, 30))) * 0.8
    return v


def game_tables(path, p):
    """-> per-step tables of side p: pos2uid[t], death[uid], dep_events[uid] = [(step, amount)], worth margin[t], has_full"""
    st = json.load(gzip.open(path))["steps"]; T = len(st); last = T - 1
    pos2uid = []; death = {}; dep = {}; wm = np.zeros(T); prev = None; has_full = True
    for t in range(T):
        o = st[t][0]["observation"]; P = o["players"][p]; W = P[2]
        full = o.get("full_carbon")
        if full is None:
            has_full = False
        wm[t] = _worth(o, full, p, last - t) - _worth(o, full, 1 - p, last - t)
        pos2uid.append({int(w[0]): (u, w[2]) for u, w in W.items()})
        bases = {int(b) for b in P[1].values()}
        if prev is not None:
            for u in prev:
                if u not in W and u not in death:
                    death[u] = t
            for u, w in W.items():
                if u in prev and w[2] == "COLLECTOR" and prev[u][1] > 5 and w[1] < 0.5 and int(w[0]) in bases:
                    dep.setdefault(u, []).append((t, float(prev[u][1])))
        prev = W
    return pos2uid, death, dep, wm, has_full, last


def one(task):
    path, p, steps, tok_pos, tok_mask, tok_kind = task
    import cnet2_feat as F
    pos2uid, death, dep, wm, has_full, last = game_tables(path, p)
    n = len(steps)
    die = np.zeros((n, 10, 3), np.uint8); die_m = np.zeros((n, 10, 3), bool)
    dp = np.zeros((n, 10, 2), np.float16); dp_m = np.zeros((n, 10, 2), bool)
    dw = np.zeros((n, 3), np.float16); dw_m = np.zeros((n, 3), bool); miss = 0
    for i, t in enumerate(steps):
        t = int(t)
        for j, k in enumerate(KD):
            if t + k <= last:
                dw[i, j] = np.clip((wm[t + k] - wm[t]) / 1000.0, -3, 3); dw_m[i, j] = True
        for s in range(10):
            if not tok_mask[i, s]:
                continue
            cp = int(tok_pos[i, s]); pos = int(F.CANON[cp]) if p else cp
            hit = pos2uid[t].get(pos)
            if hit is None:
                miss += 1; continue
            u, kind = hit
            for j, k in enumerate(KD):
                if t + k <= last:
                    die_m[i, s, j] = True; die[i, s, j] = 1 if (u in death and death[u] <= t + k) else 0
            if kind == "COLLECTOR":
                for j, k in enumerate(KP):
                    if t + k <= last:
                        dp_m[i, s, j] = True
                        dp[i, s, j] = min(sum(a for (td, a) in dep.get(u, ()) if t < td <= t + k) / 100.0, 60.0)
    return die, die_m, dp, dp_m, dw, dw_m, miss, has_full


def main():
    ds = sys.argv[1]; W = int(sys.argv[2]) if len(sys.argv) > 2 else (os.cpu_count() or 4)
    meta = json.load(open(os.path.join(ds, "meta.json")))
    gid = np.load(os.path.join(ds, "gid.npy")); step = np.load(os.path.join(ds, "step.npy"))
    tok_pos = np.load(os.path.join(ds, "tok_pos.npy"), mmap_mode="r"); tok_mask = np.load(os.path.join(ds, "tok_mask.npy"), mmap_mode="r")
    tok_kind = np.load(os.path.join(ds, "tok_kind.npy"), mmap_mode="r")
    games = {g["gi"]: g for g in meta["games"]}
    N = len(gid); tasks = []; spans = []
    i = 0
    while i < N:                                   # rows of one (game, side): contiguous, steps increasing
        g = int(gid[i]); j = i + 1
        while j < N and gid[j] == g and step[j] > step[j - 1]:
            j += 1
        k = sum(1 for (a, b, gg) in spans if gg == g)          # 0 for the first side of this game, 1 for the second
        sides = games[g]["sides"]
        if k >= len(sides):
            raise SystemExit(f"aux_labels: game {g} has more row blocks than teacher sides")
        tasks.append((games[g]["file"], sides[k], np.asarray(step[i:j]), np.asarray(tok_pos[i:j, :10]),
                      np.asarray(tok_mask[i:j, :10]), np.asarray(tok_kind[i:j, :10])))
        spans.append((i, j, g)); i = j
    out = {"aux_die": np.zeros((N, 10, 3), np.uint8), "aux_die_m": np.zeros((N, 10, 3), bool),
           "aux_dep": np.zeros((N, 10, 2), np.float16), "aux_dep_m": np.zeros((N, 10, 2), bool),
           "aux_dw": np.zeros((N, 3), np.float16), "aux_dw_m": np.zeros((N, 3), bool)}
    miss = 0; nofull = 0
    with ProcessPoolExecutor(W) as ex:
        for (a, b, _), r in zip(spans, ex.map(one, tasks, chunksize=4)):
            for nm, arr in zip(("aux_die", "aux_die_m", "aux_dep", "aux_dep_m", "aux_dw", "aux_dw_m"), r[:6]):
                out[nm][a:b] = arr
            miss += r[6]; nofull += 0 if r[7] else 1
    for nm, arr in out.items():
        np.save(os.path.join(ds, nm + ".npy"), arr)
    dm = out["aux_die_m"]; pm = out["aux_dep_m"]; wm_ = out["aux_dw_m"]
    print(json.dumps({"rows": int(N), "blocks": len(spans), "slots_unmatched": int(miss), "games_without_full_carbon": int(nofull),
                      "die_rate": [round(float(out["aux_die"][..., j][dm[..., j]].mean()), 4) for j in range(3)],
                      "dep_mean_x100": [round(float(out["aux_dep"][..., j][pm[..., j]].astype(np.float32).mean()) * 100, 1) for j in range(2)],
                      "dw_sd_x1000": [round(float(out["aux_dw"][:, j][wm_[:, j]].astype(np.float32).std()) * 1000) for j in range(3)]}), flush=True)


if __name__ == "__main__":
    main()
