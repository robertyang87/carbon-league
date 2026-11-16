"""opp_attrition -- a base pack wrapped with a scripted "attrition" behaviour (diagnostic / pressure opponent).

Ladder forensics (tools/ladder_death_forensics.py, Echo4 corpus): 92% of worker deaths are 1-for-1
mutual annihilations -- planter meets planter, or an EMPTY collector meets an EMPTY collector (tie
for least carbon) -- and the never-beaten opponents produce ~50 of them per game against ~32 for the
rest and ~23 for every member of our training pool. This wrapper lets any pack produce that
pressure on demand: after the base pack has chosen its actions, each of its tie-capable units
(planters, collectors carrying 0) that has an enemy unit of the same tie class within reach is, with
probability Q, redirected at it.

  ATTR_BASE   base pack path relative to code/ (default packs/echo4_int8.py)
  ATTR_Q      probability per eligible unit-step of taking the attrition move (default 0.5)
  ATTR_R      reach: 1 = only step onto an adjacent target's cell; 2 = also close in from distance 2
              (default 2)
  ATTR_SEED   rng seed offset (default 0); the stream is also keyed by the observation's player
              index and reset at step 0, so a game is reproducible

Recruit actions and every unit without a target in reach keep the base pack's action.
"""
import importlib.util
import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
_N = 15
_BASE = None
_RNG = {}


def _load():
    global _BASE
    if _BASE is None:
        path = os.path.join(_HERE, "..", os.environ.get("ATTR_BASE", "packs/echo4_int8.py"))
        spec = importlib.util.spec_from_file_location(f"attr_base_{os.getpid()}_{id(spec_guard)}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _BASE = mod.agent
    return _BASE


def spec_guard():      # only used for a unique module name
    return None


def _dist(a, b):
    ra, ca = divmod(a, _N); rb, cb = divmod(b, _N)
    dr = (rb - ra) % _N; dc = (cb - ca) % _N
    sr = dr if dr <= _N - dr else dr - _N          # signed shortest row / col offsets
    sc = dc if dc <= _N - dc else dc - _N
    return abs(sr) + abs(sc), sr, sc


def agent(obs, config):
    base = _load()
    acts = dict(base(obs, config) or {})
    me = obs["player"]
    if obs["step"] == 0 or me not in _RNG:
        _RNG[me] = random.Random(1000003 * int(os.environ.get("ATTR_SEED", "0")) + 7919 * me + 17)
    rng = _RNG[me]
    q = float(os.environ.get("ATTR_Q", "0.5")); reach = int(os.environ.get("ATTR_R", "2"))
    mine = obs["players"][me][2]; theirs = obs["players"][1 - me][2]
    tgt = {"PLANTER": [v[0] for v in theirs.values() if v[2] == "PLANTER"],
           "COLLECTOR": [v[0] for v in theirs.values() if v[2] == "COLLECTOR" and v[1] == 0]}
    for uid, (pos, carbon, role) in mine.items():
        if role == "COLLECTOR" and carbon != 0:
            continue
        cands = tgt.get(role) or []
        best = None
        for t in cands:
            d, sr, sc = _dist(pos, t)
            if 1 <= d <= reach and (best is None or d < best[0]):
                best = (d, sr, sc)
        if best is None or rng.random() >= q:
            continue
        d, sr, sc = best
        if sr != 0 and (sc == 0 or rng.random() < 0.5):
            acts[uid] = "DOWN" if sr > 0 else "UP"
        else:
            acts[uid] = "RIGHT" if sc > 0 else "LEFT"
    return acts
