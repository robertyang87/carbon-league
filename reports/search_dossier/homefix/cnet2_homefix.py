"""cnet2_homefix -- endgame return-home rule layer (EXIT OVERLAY form).

Registered rule: reports/REGISTRATION_COMPOSITE_RULE_LAYER_20261116.md
section 1 (Part A, pinned by the strategy group 2026-11-16). Mechanism
reference: the Master3s2 go_home hard rule, re-implemented to OUR
caliber -- not a port (no 3s2 code, no probability scan, no deadliness
check, no cargo-scaled spare window). The registration deliberately
drops those: the only pinned parameter is HOME_BUFFER.

Pinned semantics (registration section 1, verbatim caliber):

  * EPISODE_STEPS = 300 (steps 0..299); at step t the number of moves
    still available is moves_available(t) = 300 - t.
  * d = a worker's shortest-path distance to its nearest own base
    under the game's movement rules, STATIC obstacles only (dynamic
    blocking ignored). Determination for this game (verified against
    fastsim.py / cnet2_feat.py, 2026-11-16): movement is 4-directional
    on a 15x15 torus with per-coordinate wraparound (fastsim
    _MOVE_OFFSETS / _move_nbr; feat DEST); a move is NEVER blocked by
    a static feature -- trees, bases and units do not prevent the move
    itself, collisions resolve afterwards (fastsim step phases 1-2).
    The board therefore has NO static obstacles, and the shortest
    path equals the torus Manhattan distance, which is exactly
    cnet2_feat.DIST (min wrap per axis). This module reads DIST / DEST
    / MOVES from cnet2_feat so the metric is the game's own tables,
    never a re-derived copy.
  * Trigger (per own worker, re-evaluated every step): cargo > 0 AND
    moves_available(t) <= d + HOME_BUFFER, HOME_BUFFER = 2.
  * On trigger the worker's action for this step is overwritten with
    a RETURN STEP: a move whose destination has distance exactly
    d - 1. If several moves qualify, the FIRST by the pinned
    deterministic direction order wins: game action indexing order
    k = 1, 2, 3, 4 -> UP, RIGHT, DOWN, LEFT (cnet2_feat.MOVES indices;
    k = 0 is NONE/stay). No randomness, no policy-probability scan.
    If no move reduces d, the worker waits in place (NONE).
  * Unloaded workers and workers already at an own base are
    untouched (at an own base d = 0; a collector standing there has
    already deposited -- fastsim phase 3 zeroes its cargo -- and the
    d == 0 case is skipped explicitly as well). Both seats: the rule
    reads obs["player"], so it applies symmetrically.
  * Form: EXIT OVERLAY only -- apply_homefix is called on the final
    joint action the search agent returns (all exit paths of the
    constructed agent in cnet2_search). Candidate generation and the
    comparator never see the rule.

The decision is a deterministic pure function of (obs, acts): no RNG,
no state carried between calls. The module-level LAST dict is a
diagnostic side channel only (per-call check/trigger record for the
probe harness); it is overwritten on every call and never read by the
rule itself.
"""

import cnet2_feat as F  # noqa: E402  (game geometry tables: DIST / DEST / MOVES)

EPISODE_STEPS = 300
HOME_BUFFER = 2
# Pinned deterministic direction order: game action indexing order
# (cnet2_feat.MOVES indices 1..4) -- UP, RIGHT, DOWN, LEFT.
DIRECTION_ORDER = (1, 2, 3, 4)

LAST = {"checked": 0, "triggered": []}


def _dist_to_bases(pos, bases):
    return min(int(F.DIST[pos, b]) for b in bases)


def _return_move(pos, bases, d):
    """First move (pinned direction order) whose destination is at
    distance exactly d - 1 from the nearest own base; None if no move
    reduces d. (On this board's metric a reducing move always exists
    when d > 0 -- DIST is the graph distance of the torus grid and
    DEST lists its edges -- so None is a defensive fallback only.)"""
    for k in DIRECTION_ORDER:
        dest = int(F.DEST[pos, k])
        if _dist_to_bases(dest, bases) == d - 1:
            return F.MOVES[k]
    return None


def apply_homefix(obs, acts):
    """Return the final joint action with return-home overrides
    applied for every triggering own worker. `acts` is not mutated;
    when nothing triggers, `acts` itself is returned unchanged."""
    LAST["checked"] = 0
    LAST["triggered"] = []
    me = obs["player"]
    moves_available = EPISODE_STEPS - int(obs["step"])
    players = obs["players"]
    bases = [int(b) for b in players[me][1].values()]
    if not bases:
        return acts
    out = None
    for uid, w in players[me][2].items():
        cargo = float(w[1])
        if cargo <= 0:
            continue
        LAST["checked"] += 1
        pos = int(w[0])
        d = _dist_to_bases(pos, bases)
        if d == 0:
            continue
        if moves_available > d + HOME_BUFFER:
            continue
        move = _return_move(pos, bases, d)
        chosen = move if move is not None else "NONE"
        if out is None:
            out = dict(acts)
        prev = out.get(uid, "NONE")
        out[uid] = chosen
        LAST["triggered"].append({
            "uid": uid, "step": int(obs["step"]), "cargo": cargo,
            "d": d, "moves_available": moves_available,
            "move": chosen, "prev": prev, "changed": chosen != prev,
        })
    return out if out is not None else acts
