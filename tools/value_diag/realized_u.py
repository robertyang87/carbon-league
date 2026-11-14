#!/usr/bin/env python3
"""Realized U per root (PREREG_VALUE 'realized U' definition).

Continuation definition (signed preregistration, verbatim in
substance): the truth value is the FINAL outcome of the source game
as actually played and recorded -- the source replay played out to
its end by the original play, NOT a re-play and NOT a continuation
by any substituted policy. I.e., U_real(root) = U of the terminal
margin of the recorded game, from the seat of the root.

Source field chain (see PROVENANCE.md):
  ds corpus index final scores -> ds_m3s2 meta games table 'scores'
  = [score_player0, score_player1] -> carried per game into Stage 1's
  rootset32.json ('scores'). These are exactly the values
  cnet2_build.build_game receives as its per-side 'margins' argument
  (margin_side = scores[side] - scores[1-side]) and cross-checks
  against the raw exact-label payload when present; the build
  product's own u_raw column for normal games IS
  cnet2_build._u_of_margin(margin_raw).

Formula (identical to cnet2_build._u_of_margin and to the platform
Terminal-U rule):  U = 0.5*W + 0.5*sigmoid(margin/1000),
  W = 1 / 0.5 / 0 for margin > / = / < 0.

Self-checks (all hard-asserted; any failure aborts):
  1. every (game, seat): all its roots share one identical U;
  2. per game: U(seat0) + U(seat1) == 1 (float64, 1e-12), and
     sign(U0 - 0.5) == sign(scores[0] - scores[1]) (both 0 iff draw);
  3. per (game, seat): exactly 99 roots, steps within [200, 298] and
     <= the game's final recorded step;
  4. U == cnet2_build._u_of_margin(margin_seat) for every (game, seat)
     (the build product's own formula function, imported live).
"""
import gzip
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, VD, OUT, load_roots  # noqa: E402


def u_of_margin(m):
    w = 1.0 if m > 0 else (0.0 if m < 0 else 0.5)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-float(m) / 1000.0))


def main():
    os.makedirs(OUT, exist_ok=True)
    rs = json.load(open(os.path.join(WORK, "rootset32.json")))
    # SHAPE FIX (2026-11-14, step 1 crash): rootset32.json is a bare
    # LIST of 32 game dicts (keys: key/md5/names/sides_in_ds/gi/scores)
    # -- the Stage 1 writer never wrapped it in {"games": [...]}.
    # The crash was purely a reader-shape mismatch (data md5 already
    # verified at step 0). Entries are keyed by g["key"] = replay
    # filename, which is exactly what build_roots.py wrote into
    # roots.npz game_key. Measurement definition unchanged.
    assert isinstance(rs, list) and len(rs) == 32, type(rs)
    games = rs
    z = load_roots()
    gk = np.asarray([str(x) for x in z["game_key"]])
    seat = np.asarray(z["seat"]).astype(int)
    step = np.asarray(z["step_arr"]).astype(int)
    n = len(gk)
    assert n == 6336, n
    name2gi = {g["key"]: i for i, g in enumerate(games)}
    gidx = np.array([name2gi[x] for x in gk], dtype=int)

    # per-game final recorded step from the source replays themselves
    total_steps = {}
    for g in games:
        with gzip.open(os.path.join(WORK, "roots", g["key"]), "rt") as f:
            d = json.load(f)
        total_steps[g["key"]] = len(d["steps"])

    sys.path.insert(0, "/root/carbon/code")
    from cnet2_build import _u_of_margin as build_u

    U = np.zeros(n, np.float64)
    margin = np.zeros(n, np.float64)
    dist = np.zeros(n, np.int32)   # steps from this root to game end
    per_game = []
    for gi, g in enumerate(games):
        s0, s1 = float(g["scores"][0]), float(g["scores"][1])
        final_step = total_steps[g["key"]] - 1
        us = {}
        for sd in (0, 1):
            m = (s0 - s1) if sd == 0 else (s1 - s0)
            u = u_of_margin(m)
            assert abs(u - build_u(m)) < 1e-12, (g["key"], sd, u, build_u(m))
            us[sd] = (m, u)
            sel = (gidx == gi) & (seat == sd)
            assert int(sel.sum()) == 99, (g["key"], sd, int(sel.sum()))
            assert step[sel].min() >= 200 and step[sel].max() <= 298
            assert step[sel].max() <= final_step
            U[sel] = u
            margin[sel] = m
            dist[sel] = final_step - step[sel]
            assert len(set(U[sel].tolist())) == 1          # check 1
        u0, u1 = us[0][1], us[1][1]
        assert abs(u0 + u1 - 1.0) < 1e-12, (g["key"], u0, u1)   # check 2
        d = (s0 > s1) - (s0 < s1)
        assert (u0 > 0.5) - (u0 < 0.5) == d, (g["key"], u0, d)
        per_game.append({
            "name": g["key"], "scores": [s0, s1],
            "margin_seat0": us[0][0], "margin_seat1": us[1][0],
            "U_seat0": u0, "U_seat1": u1,
            "total_steps": total_steps[g["key"]],
            "final_step": final_step,
        })

    np.savez(os.path.join(OUT, "realized_u.npz"),
             U=U, margin=margin, dist_to_end=dist,
             game_idx=gidx, seat=seat, step=step)
    prov = {
        "definition": ("truth = the source game's recorded actual play "
                       "to its end (no re-play, no policy substitution); "
                       "seat-oriented terminal margin -> platform U"),
        "source_field": ("rootset32.json games[].scores (ds_m3s2 build "
                         "meta games table; the same values build_game "
                         "receives as per-side margins)"),
        "formula": "U = 0.5*W + 0.5*sigmoid(margin/1000), W=1/0.5/0",
        "formula_crosscheck": ("equals cnet2_build._u_of_margin for "
                               "every (game, seat), asserted live"),
        "orientation": "margin_seat = scores[seat] - scores[1-seat]",
        "n_roots": int(n),
        "games": per_game,
    }
    with open(os.path.join(OUT, "realized_u_provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    print("realized_u: OK n=%d games=%d (self-checks 1-4 passed)"
          % (n, len(games)))


if __name__ == "__main__":
    main()
