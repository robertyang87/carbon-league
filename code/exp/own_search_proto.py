"""own_search_proto -- minimal joint-move rollout search on OUR actor. No external strategy code.

Each turn:
  0. the base pack (OWN_BASE, default packs/echo4_int8.py) plays its move -> candidate 0, always kept;
  1. K-1 more whole-turn joint moves are SAMPLED from the same actor's policy (temperatures OWN_TEMPS);
  2. the opponent's root move is sampled once from the same net seen from its seat and SHARED by all
     candidates (common random numbers); later rollout steps share one noise row across candidates too;
  3. every candidate is played forward in FastCarbon, both sides moving by the rollout policy
     (OWN_ROLL: "self" = the actor itself, or a path to a smaller npz such as ../data/weights/cnet2_s64.npz);
  4. staged elimination: after 1 step keep the best half, after 4 steps the best quarter (candidate 0
     always survives), survivors go on to OWN_DEPTH steps or until the time budget (OWN_BUDGET seconds
     per call, base pack included) would be exceeded;
  5. leaf = heuristic worth margin (cash + 0.9 cargo + worker value + tree value; the exact cash margin
     once the game is over); switch only if the best survivor beats candidate 0 by more than OWN_EPS.

Env: OWN_K (8) OWN_DEPTH (8) OWN_BUDGET (0.80) OWN_EPS (20) OWN_TEMPS ("1.0,2.0,3.0")
     OWN_OPP_T (0.5) OWN_ROLL_T (0.5) OWN_TEACHER (2) OWN_ROLL_TEACHER (1 for a foreign roll net)
     OWN_SEARCH=0 -> return the base pack's move untouched (control)
     OWN_STATS=<path> -> append one JSON line of per-game search statistics after the last turn.
Prototype for measurement: the leaf is the hand-written one, there is no learned critic yet.
"""
import base64
import importlib.util
import json
import os
import sys
import time
import zlib

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.abspath(os.path.join(_HERE, ".."))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

import cnet2_feat as F  # noqa: E402
import cnet2_search as S  # noqa: E402
from fastsim import FastCarbon  # noqa: E402


def _env(k, d, cast=float):
    return cast(os.environ.get(k, d))


_spec = importlib.util.spec_from_file_location(f"own_base_{os.getpid()}",
                                               os.path.join(_CODE, os.environ.get("OWN_BASE", "packs/echo4_int8.py")))
_B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_B)
_w = _B._dequantize(zlib.decompress(base64.b64decode(_B._B1)), _B._Q1)
_w["meta"] = np.array(_B._M1)
_POL = S.CachePolicy(S.load_weights(_w))
_T = _env("OWN_TEACHER", "2", int)
_roll = os.environ.get("OWN_ROLL", "self")
if _roll == "self":
    _RPOL, _RT = _POL, _T
else:
    _RPOL = S.CachePolicy(S.load_weights(os.path.join(_HERE, _roll) if not os.path.isabs(_roll) else _roll))
    _RT = _env("OWN_ROLL_TEACHER", "1", int)

_ST = {}
STATS = {}


def _reset_stats():
    STATS.clear()
    STATS.update(turns=0, searched=0, changed=0, cands=0, depth={}, time=0.0, worst=0.0, errors=0, cut=0)


def _leaf(sim, me, left):
    if sim.step_count >= 299 or left <= 0:
        return float(sim.cash[me] - sim.cash[1 - me])
    v = [float(sim.cash[0]), float(sim.cash[1])]
    wv = 30.0 * min(1.0, left / 100.0)
    for w in sim.workers.values():
        v[w.player] += wv + (0.9 * float(w.carbon) if w.is_collector else 0.0)
    for t in sim.trees.values():
        v[t.player] += 2.0 * max(0.0, min(50.0 - t.age, left))
    return v[me] - v[1 - me]


def _decode(pol, teacher, xs, auxs, noise=None, temp=None, forced=None):
    X = S.stack_x(xs)
    tok, ctx = S.encode_b(pol, X, np.full(len(xs), teacher))
    return S.decode_b(pol, X, auxs, tok, ctx, forced=forced, noise=noise, temp=temp)


def _opp_obs(obs, me, f):
    o = dict(obs); o["player"] = 1 - me
    vis = np.zeros(F.N)
    for wv in obs["players"][1 - me][2].values():
        cs = [wv[0]] if wv[2] == "COLLECTOR" else F.DEST[wv[0], 1:]
        vis[cs] = f.est[cs]
    o["carbon"] = vis.tolist()
    return o


def _search(obs, act0, f, fo, x, xo, t0, rng):
    me = obs["player"]; step = obs["step"]; left = 299 - step
    K = _env("OWN_K", "8", int); D = min(_env("OWN_DEPTH", "8", int), left)
    budget = _env("OWN_BUDGET", "0.80"); eps = _env("OWN_EPS", "20")
    temps = [float(v) for v in os.environ.get("OWN_TEMPS", "1.0,2.0,3.0").split(",")]
    if D <= 0 or K < 2:
        return act0
    # ---- candidates: 0 = the base pack's move; the rest sampled from the actor
    y0 = f.labels(act0)
    n = K - 1
    X1 = S.stack_x([x])
    tok1, ctx1 = S.encode_b(_POL, X1, np.full(1, _T))          # the trunk runs ONCE for the root; samples only re-decode
    Xn = S.stack_x([x] * n)
    ys = S.decode_b(_POL, Xn, [f.aux] * n, np.repeat(tok1, n, 0), np.repeat(ctx1, n, 0),
                    noise=rng.random((n, F.OWN)), temp=np.array([temps[i % len(temps)] for i in range(n)]))
    cands, seen = [y0], {tuple(int(v) for v in y0)}
    for y in ys:
        k = tuple(int(v) for v in y)
        if k not in seen:
            seen.add(k); cands.append(y)
    STATS["cands"] += len(cands)
    if len(cands) < 2:
        return act0
    # ---- opponent root move, shared by every candidate
    yo = _decode(_POL, _T, [xo], [fo.aux], noise=rng.random((1, F.OWN)),
                 temp=np.array([_env("OWN_OPP_T", "0.5")]))[0]
    ao = fo.to_actions(yo)
    full = {"step": step, "players": obs["players"], "trees": obs.get("trees") or {},
            "full_carbon": [float(v) for v in f.est], "carbon": obs["carbon"], "remainingOverageTime": 60}
    br = []
    for y in cands:
        sim = FastCarbon(); sim.load_observation(full)
        am = f.to_actions(y)
        sim.step(*((am, ao) if me == 0 else (ao, am)))
        br.append({"y": y, "sim": sim, "fm": S.feat_copy(f), "fo": S.feat_copy(fo)})
    alive = list(range(len(br))); depth = 1
    val = np.array([_leaf(b["sim"], me, left - 1) for b in br])
    roll_t = _env("OWN_ROLL_T", "0.5")
    t_step = None
    while depth < D:
        if depth == 1 and len(alive) > 2:
            keep = max(2, (len(alive) + 1) // 2)
        elif depth == 4 and len(alive) > 2:
            keep = max(2, (len(br) + 3) // 4)
        else:
            keep = len(alive)
        if keep < len(alive):
            order = sorted(alive, key=lambda i: -val[i])[:keep]
            if 0 not in order:
                order[-1] = 0
            alive = sorted(order)
        now = time.perf_counter() - t0
        est = (t_step * len(alive) if t_step is not None else 0.03 * len(alive))
        if now + 1.3 * est > budget:
            STATS["cut"] += 1
            break
        ts = time.perf_counter()
        live = [i for i in alive if br[i]["sim"].step_count < 299]
        if live:
            xm = [br[i]["fm"].observe(br[i]["sim"].obs(me)) for i in live]
            xq = [br[i]["fo"].observe(br[i]["sim"].obs(1 - me)) for i in live]
            nz = np.tile(rng.random((1, F.OWN)), (len(live), 1)); nq = np.tile(rng.random((1, F.OWN)), (len(live), 1))
            tt = np.full(len(live), roll_t)
            ym = _decode(_RPOL, _RT, xm, [br[i]["fm"].aux for i in live], noise=nz, temp=tt)
            yq = _decode(_RPOL, _RT, xq, [br[i]["fo"].aux for i in live], noise=nq, temp=tt)
            for j, i in enumerate(live):
                b = br[i]
                a1 = b["fm"].to_actions(ym[j]); a2 = b["fo"].to_actions(yq[j])
                b["sim"].step(*((a1, a2) if me == 0 else (a2, a1)))
        depth += 1
        for i in alive:
            val[i] = _leaf(br[i]["sim"], me, left - depth)
        t_step = (time.perf_counter() - ts) / max(1, len(alive))
    STATS["searched"] += 1
    STATS["depth"][depth] = STATS["depth"].get(depth, 0) + 1
    best = max(alive, key=lambda i: val[i])
    if best != 0 and val[best] - val[0] > eps:
        STATS["changed"] += 1
        return f.to_actions(br[best]["y"])
    return act0


def agent(obs, config=None):
    t0 = time.perf_counter()
    me = obs["player"]; step = obs["step"]
    act0 = dict(_B.agent(obs, config) or {})
    if os.environ.get("OWN_SEARCH", "1") == "0":
        return act0
    st = _ST.get(me)
    if step == 0 or st is None:
        st = _ST[me] = {"f": F.Feat(me), "fo": F.Feat(1 - me), "rng": np.random.default_rng(12345 + me)}
        _reset_stats()
    out = act0
    try:
        x = st["f"].observe(obs)
        xo = st["fo"].observe(_opp_obs(obs, me, st["f"]))
        STATS["turns"] += 1
        out = _search(obs, act0, st["f"], st["fo"], x, xo, t0, st["rng"])
    except Exception as e:
        STATS["errors"] = STATS.get("errors", 0) + 1; STATS["last_error"] = repr(e)[:300]
        if os.environ.get("CARBON_DEBUG"):
            raise
    dt = time.perf_counter() - t0
    STATS["time"] += dt; STATS["worst"] = max(STATS["worst"], dt)
    if step >= 298 and os.environ.get("OWN_STATS"):
        with open(os.environ["OWN_STATS"], "a") as fh:
            fh.write(json.dumps(dict(STATS, player=me, mean_s=round(STATS["time"] / max(1, STATS["turns"]), 3))) + "\n")
    return out
