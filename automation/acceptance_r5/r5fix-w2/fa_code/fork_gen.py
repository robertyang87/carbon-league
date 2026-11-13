"""fork_gen -- value of ONE decision, measured from the same position (credit assignment without whole-game noise).

Whole-game outcome weighting drowns a recruit / plant decision: two games that differ in one early move are 185 points
apart after 25 steps, 360 after 50 and about 1000 at the end, while the decision itself is worth tens of points.
Here one greedy game is played (our pure net against an opponent, both deterministic). At randomly chosen steps the
process forks: the child replaces exactly one of our decisions at that step, then both sides play on greedily for up to
HMAX steps; the parent plays the unchanged game. Nothing else differs, so
    effect(h) = position value of the child - position value of the parent, h steps after the decision.
Position value (ours minus theirs): cash + cargo on board + remaining value of standing trees (see _value). At the last
step it is the final score difference.

Decisions that are forked (kind):
  base slot   P>C  P>0  C>P  C>0  0>P  0>C     (recruit planter / collector / nothing; recruits only with cash >= 30)
  collector   car>2nd      plays the net's second choice instead of its first (records p1, p2, cargo, a1, a2);
                           needs an agent that exposes .inner (awr_us_env.py)
  planter     plant>move   it would plant here (stays on a free cell, cash >= price): move to a random neighbour instead
              move>plant   it stands on a plantable cell and moves away: stay (plant) instead

  usage: fork_gen.py OUT_DIR OUR_AGENT.py OPP_AGENT.py SEED0 NSEEDS [WORKERS]
  env:   FORK_P (0.15)      chance per step to fork one decision
         FORK_H (10,25,50)  horizons recorded; the child stops after the largest
         FORK_END (0.0)     share of forks whose child plays to the end of the game (records the final difference too)
         FORK_KINDS         only these kinds, e.g. "P>C,C>P,plant>move"
         FORK_TAG           prefix for the output file name
Output: OUT_DIR/forks_<tag><seed0>.jsonl, one line per fork:
  {seed, side, t, kind, unit, cash, n_pl, n_car, n_tree, e_tree, price, d: {h: effect}, same: true if the child's game
   never differed from the parent's (the engine ignored the change)}
Linux only (os.fork). One BLAS thread per process."""
import json
import os
import random
import sys

import numpy as np

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MOVES = ("UP", "DOWN", "LEFT", "RIGHT")
CAR_MOVES = ["NONE", "UP", "RIGHT", "DOWN", "LEFT"]          # slot order of the net's 5 logits (cnet2_feat.MOVES)
T_END = 299


def _value(sim, p, idx):
    """position value of player p minus the opponent's; idx = number of steps played"""
    st = sim.get_state()
    if idx >= T_END:
        return float(st["cash"][p]) - float(st["cash"][1 - p])
    left = T_END - idx
    v = [0.0, 0.0]
    for q in (0, 1):
        v[q] = float(st["cash"][q]) + sum(float(w["carbon"]) for w in st["workers"].values() if w["player"] == q)
    epl = {q: [int(w["pos"]) for w in st["workers"].values() if w["player"] != q and w["type"] == "PLANTER"] for q in (0, 1)}
    for t_ in st["trees"].values():
        q = t_["player"]; pos = int(t_["pos"]); r, c = divmod(pos, 15)
        nb = sum(sim.carbon[((r + dr) % 15) * 15 + (c + dc) % 15] for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        life = max(0, min(50 - int(t_["age"]), left, 30))
        near = any(min(abs(e // 15 - r), 15 - abs(e // 15 - r)) + min(abs(e % 15 - c), 15 - abs(e % 15 - c)) <= 1 for e in epl[q])
        v[q] += nb * (1.0 - 0.9625 ** life) * (0.3 if near else 0.8)
    return v[p] - v[1 - p]


def _alts(obs, acts, rng, kinds, limit=10, inner=None):
    me = obs["player"]; P = obs["players"]; cash = float(P[me][0]); out = []
    room = len(P[me][2]) < limit                       # the engine ignores a recruit at the worker limit
    trees = [v for q in (0, 1) for v in (P[q][3].values() if isinstance(P[q][3], dict) else [])]
    tcells = {int(v[0]) for v in trees}; bcells = {int(b) for q in (0, 1) for b in P[q][1].values()}
    mine = len(P[me][3]) if isinstance(P[me][3], dict) else 0
    price = 20.0 if mine <= 0 else 20.0 + 5.0 * 1.235 ** sum(1 for v in trees if int(v[1]) < 50)
    for b in P[me][1]:
        a = acts.get(b)
        cur = "P" if a == "RECPLANTER" else "C" if a == "RECCOLLECTOR" else "0"
        for new, act in (("P", "RECPLANTER"), ("C", "RECCOLLECTOR"), ("0", None)):
            if new != cur and (new == "0" or (cash >= 30 and room)) and (cur == "0" or room):
                out.append((f"{cur}>{new}", b, act))
    for u, w in P[me][2].items():
        if w[2] != "PLANTER":
            continue
        pos = int(w[0]); a = acts.get(u)
        if pos in tcells or pos in bcells or cash < price:
            continue
        if a not in MOVES:
            out.append(("plant>move", u, rng.choice(MOVES)))
        else:
            out.append(("move>plant", u, None))
    if inner is not None and (not kinds or "car>2nd" in kinds):
        # a collector plays the net's SECOND choice instead of its first (how good is the net's ranking of car moves?)
        try:
            lg = np.asarray(inner.state["lg0"], dtype=np.float64); uids = list(inner.state["f"].own_uid[:10])
            for s_, u in enumerate(uids):
                w = P[me][2].get(u) if u is not None else None
                if w is None or w[2] != "COLLECTOR":
                    continue
                played = acts.get(u, "NONE")
                if played not in CAR_MOVES:
                    continue
                z = lg[s_][:5]; z = z - np.max(z[np.isfinite(z)]); pr = np.where(np.isfinite(z), np.exp(z), 0.0); pr = pr / pr.sum()
                i0 = CAR_MOVES.index(played)
                others = [i for i in np.argsort(-pr) if i != i0 and pr[i] > 0]
                if others:
                    i1 = int(others[0])
                    out.append(("car>2nd", u, None if CAR_MOVES[i1] == "NONE" else CAR_MOVES[i1],
                                dict(p1=round(float(pr[i0]), 4), p2=round(float(pr[i1]), 4), cargo=round(float(w[1]), 1),
                                     a1=played, a2=CAR_MOVES[i1])))
        except Exception:
            pass
    if kinds:
        out = [x for x in out if x[0] in kinds]
    return out, price


def play(task):
    out, us, opp, seed, p_fork, hs, p_end, kinds = task
    import eval_v53 as E
    from fastsim import FastCarbon
    rng = random.Random(seed * 7919 + 13)
    side = seed % 2
    sim = FastCarbon(E.CFG).reset(seed=seed)
    agents = [None, None]
    agents[side] = E.load_agent_fn(os.path.join(HERE, us))
    agents[1 - side] = E.load_agent_fn(os.path.join(HERE, opp))
    V = []; forks = []; hmax = max(hs)
    for t in range(T_END):
        obs = [sim.obs(0), sim.obs(1)]
        acts = [agents[q](obs[q], E.CFG) or {} for q in (0, 1)]
        V.append(_value(sim, side, t))
        if rng.random() < p_fork and t < T_END - min(hs):
            cand, price = _alts(obs[side], acts[side], rng, kinds, int(E.CFG.get("workerLimit", 10)), getattr(agents[side], "inner", None))
            if cand:
                kk = rng.choice(sorted({c[0] for c in cand}))          # a kind first, then one decision of that kind
                ch = rng.choice([c for c in cand if c[0] == kk])
                kind, unit, act = ch[:3]; extra = ch[3] if len(ch) > 3 else {}
                to_end = rng.random() < p_end
                Pm = obs[side]["players"][side]; Po = obs[side]["players"][1 - side]
                rec = dict(seed=seed, side=side, t=t, kind=kind, unit=unit, cash=round(float(Pm[0]), 1), price=round(price, 1),
                           n_pl=sum(1 for w in Pm[2].values() if w[2] == "PLANTER"), n_car=sum(1 for w in Pm[2].values() if w[2] == "COLLECTOR"),
                           n_tree=len(Pm[3]), e_tree=len(Po[3]), e_pl=sum(1 for w in Po[2].values() if w[2] == "PLANTER"), to_end=to_end)
                rec.update(extra)
                r, w = os.pipe(); pid = os.fork()
                if pid == 0:                                   # child: the changed game
                    try:
                        os.close(r)
                        a2 = dict(acts[side])
                        if act is None:
                            a2.pop(unit, None)
                        else:
                            a2[unit] = act
                        A = [None, None]; A[side] = a2; A[1 - side] = acts[1 - side]
                        sim.step(A[0], A[1]); idx = t + 1; res = {}
                        while True:
                            h = idx - t
                            if h in hs or idx >= T_END:
                                res["end" if idx >= T_END else str(h)] = _value(sim, side, idx)
                            if idx >= T_END or (h >= hmax and not to_end):
                                break
                            o = [sim.obs(0), sim.obs(1)]
                            a = [agents[q](o[q], E.CFG) or {} for q in (0, 1)]
                            sim.step(a[0], a[1]); idx += 1
                        os.write(w, json.dumps(res).encode())
                    finally:
                        os._exit(0)
                os.close(w); buf = b""
                while True:
                    ch = os.read(r, 65536)
                    if not ch:
                        break
                    buf += ch
                os.close(r); os.waitpid(pid, 0)
                if buf:
                    rec["child"] = json.loads(buf); forks.append(rec)
        sim.step(acts[0], acts[1])
    V.append(_value(sim, side, T_END))
    rows = []
    for f in forks:
        d = {}
        for k, v in f.pop("child").items():
            idx = T_END if k == "end" else f["t"] + int(k)
            d[k] = round(v - V[idx], 2)
        f["d"] = d; f["same"] = all(abs(x) < 1e-9 for x in d.values()); f["final_margin"] = round(V[T_END], 1)
        rows.append(f)
    return rows


def main():
    out, us, opp, s0, ns = sys.argv[1:6]
    W = int(sys.argv[6]) if len(sys.argv) > 6 else (os.cpu_count() or 4)
    s0, ns = int(s0), int(ns)
    hs = tuple(int(x) for x in os.environ.get("FORK_H", "10,25,50").split(","))
    kinds = tuple(k for k in os.environ.get("FORK_KINDS", "").split(",") if k) or None
    os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, f"forks_{os.environ.get('FORK_TAG', '')}{s0}.jsonl")
    done = set()
    if os.path.exists(fn):
        done = {json.loads(l)["seed"] for l in open(fn) if l.strip()}
    tasks = [(out, us, opp, s, float(os.environ.get("FORK_P", "0.15")), hs, float(os.environ.get("FORK_END", "0")), kinds)
             for s in range(s0, s0 + ns) if s not in done]
    n = 0
    with ProcessPoolExecutor(W) as ex, open(fn, "a") as f:
        for rows in ex.map(play, tasks, chunksize=1):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush(); n += len(rows)
    print(json.dumps({"file": fn, "games": len(tasks), "forks": n}), flush=True)


if __name__ == "__main__":
    main()
