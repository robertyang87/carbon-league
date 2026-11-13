"""S1-2 A2: teacher _terminal_u == eval_v53 terminal scoring.

Part 1: 3 hand-computed cases (extinction win / extinction loss / time-out).
Part 2: 20 real terminal states, replayed from m3s2 recorded actions
(reconstruction fidelity checked against the replay's final cash first).
The eval_v53 reference formula is re-implemented here from eval_v53.py
L493-494 + _extinct (independent of stage0_teacher), and _extinct_state is
checked against eval_v53._extinct and rl2_ppo_rollout._extinct on sampled
states along the replays.
Run: <venv python> test_s12_a2.py  (CPU only)
"""
import gzip
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from eval_v53 import CFG, EPISODE_STEPS  # noqa: E402
from eval_v53 import _extinct as ev53_extinct  # noqa: E402
from fastsim import FastCarbon  # noqa: E402
import stage0_teacher as T  # noqa: E402

# rl2_ppo_rollout._extinct, exec'd standalone (importing the module would pull
# the whole adapter chain); the function only uses CFG.
import ast as _ast


def _load_rollout_extinct():
    src = open(os.path.join(HERE, "rl2_ppo_rollout.py")).read()
    tree = _ast.parse(src)
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name == "_extinct":
            ns = {"CFG": CFG}
            exec(compile(_ast.Module(body=[node], type_ignores=[]), "rollout_extinct", "exec"), ns)
            return ns["_extinct"]
    raise RuntimeError("rl2_ppo_rollout._extinct not found")


rollout_extinct = _load_rollout_extinct()

REPLAY_DIR = os.path.join(HERE, "replays")


def ev53_terminal_u(post, pre, steps_played, me):
    """Independent copy of eval_v53.py L493-494 + _extinct."""
    extinct = [ev53_extinct(post, 0), ev53_extinct(post, 1)]
    cash_fin = [float(post["cash"][0]), float(post["cash"][1])]
    pre_cash = [float(pre["cash"][0]), float(pre["cash"][1])]
    score = [(steps_played - EPISODE_STEPS - 1) if extinct[p]
             else (pre_cash[p] if any(extinct) else cash_fin[p])
             for p in range(2)]
    mg = score[me] - score[1 - me]
    w = 1.0 if mg > 0 else (0.0 if mg < 0 else 0.5)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-mg / 1000.0)), mg, extinct


class FakeSim:
    """Minimal stand-in for _terminal_u: only get_state() + step_count."""

    def __init__(self, post, step_count):
        self._post = post
        self.step_count = step_count

    def get_state(self):
        return self._post


def hand_case(name, post, pre, steps, me, expect_mg):
    u_t = T._terminal_u(FakeSim(post, steps), me, pre, episode_steps=300)
    u_r, mg, ext = ev53_terminal_u(post, pre, steps, me)
    assert mg == expect_mg, f"{name}: margin {mg} != hand {expect_mg}"
    d = abs(u_t - u_r)
    print(f"{name}: teacher U={u_t:.12f} eval_v53 U={u_r:.12f} diff={d:.2e} "
          f"margin={mg} extinct={ext}")
    assert d <= 1e-9, f"{name}: terminal U mismatch {d}"
    return d


def base_post(cash0=100.0, cash1=80.0, workers0=("w0",), workers1=("w1",), trees0=(), trees1=(),
              bases0=(0,), bases1=(1,)):
    def ws(names, p):
        return {n: {"player": p} for n in names}
    def ts(names, p):
        return {n: {"player": p} for n in names}
    return {"workers": {**ws(workers0, 0), **ws(workers1, 1)},
            "trees": {**ts(trees0, 0), **ts(trees1, 1)},
            "bases": {0: list(bases0), 1: list(bases1)},
            "cash": [cash0, cash1]}


def main():
    worst = 0.0
    # --- Part 1: hand cases. me=0 throughout. ---
    # (a) extinction win: p1 extinct at step 150 (post: no workers/trees/bases, cash 5),
    #     p0 survivor: pre cash 500, post cash 480 (recruit on killing step must NOT count).
    post = base_post(cash0=480.0, cash1=5.0, workers0=("w0",), workers1=(), trees1=(), bases1=())
    pre = base_post(cash0=500.0, cash1=40.0, workers0=("w0",), workers1=("w1",), bases1=(1,))
    # score0 = pre 500; score1 = 150-300-1 = -151; margin 651
    worst = max(worst, hand_case("extinct-win", post, pre, 150, 0, 651.0))
    # (b) extinction loss: p0 extinct at step 200, p1 survivor pre cash 300.
    post = base_post(cash0=10.0, cash1=320.0, workers0=(), workers1=("w1",), trees0=(), bases0=())
    pre = base_post(cash0=60.0, cash1=300.0, workers0=("w0",), workers1=("w1",), bases0=(0,))
    # score0 = 200-300-1 = -101; score1 = pre 300; margin -401
    worst = max(worst, hand_case("extinct-loss", post, pre, 200, 0, -401.0))
    # (c) time-out: no extinction, step 299, final cash decides.
    post = base_post(cash0=9000.0, cash1=8500.0)
    pre = base_post(cash0=8900.0, cash1=8400.0)
    worst = max(worst, hand_case("timeout", post, pre, 299, 0, 500.0))

    # --- Part 2: 20 real terminal states from m3s2 replays. ---
    files = sorted(os.path.join(REPLAY_DIR, f) for f in os.listdir(REPLAY_DIR)
                   if f.endswith(".json.gz"))[:20]
    assert len(files) == 20, f"need 20 replays, have {len(files)}"
    n_checked = 0
    extinct_wins = extinct_losses = timeouts = 0
    for rp in files:
        d = json.load(gzip.open(rp))
        st = d["steps"]
        sim = FastCarbon(CFG).reset(seed=1, full_carbon=st[0][0]["observation"]["full_carbon"])
        pre = sim.get_state()
        for t in range(299):
            pre = sim.get_state()
            sim.step(st[t + 1][0].get("action") or {}, st[t + 1][1].get("action") or {})
        post = sim.get_state()
        # fidelity: reconstructed final cash == replay final cash
        rc = [st[299][0]["observation"]["players"][0][0],
              st[299][0]["observation"]["players"][1][0]]
        assert post["cash"] == rc, f"reconstruction drift in {os.path.basename(rp)}"
        for me in (0, 1):
            u_t = T._terminal_u(sim, me, pre, episode_steps=EPISODE_STEPS)
            u_r, mg, ext = ev53_terminal_u(post, pre, sim.step_count, me)
            assert abs(u_t - u_r) <= 1e-9, f"terminal mismatch {os.path.basename(rp)} me={me}"
            worst = max(worst, abs(u_t - u_r))
            n_checked += 1
        if ev53_extinct(post, 0) or ev53_extinct(post, 1):
            if ev53_extinct(post, 1):
                extinct_wins += 1
            else:
                extinct_losses += 1
        else:
            timeouts += 1
        # _extinct mirror check on sampled states along this game
        # (re-run cheap: reuse this game's states is overkill; check final only here,
        #  mid-game states checked in the loop below via a second pass on 5 games)
    print(f"real terminals: {len(files)} games x me=0/1 = {n_checked} checks, "
          f"extinct-win {extinct_wins} / extinct-loss {extinct_losses} / timeout {timeouts}")
    print(f"worst |teacher - eval_v53| over all cases = {worst:.2e}  (tol 1e-9)")

    # _extinct mirror: teacher vs eval_v53 vs rollout on mid-game states
    n_states = 0
    n_mismatch = 0
    for rp in files[:5]:
        d = json.load(gzip.open(rp))
        st = d["steps"]
        sim = FastCarbon(CFG).reset(seed=1, full_carbon=st[0][0]["observation"]["full_carbon"])
        for t in range(299):
            if t % 25 == 0:
                s = sim.get_state()
                for p in (0, 1):
                    a = T._extinct_state(s, p)
                    b = ev53_extinct(s, p)
                    c = rollout_extinct(s, p)
                    n_states += 1
                    if not (a == b == c):
                        n_mismatch += 1
                        print(f"  MISMATCH {os.path.basename(rp)} t={t} p={p}: "
                              f"teacher={a} eval_v53={b} rollout={c}")
            sim.step(st[t + 1][0].get("action") or {}, st[t + 1][1].get("action") or {})
    print(f"_extinct mirror: {n_states} state/seat checks, mismatches={n_mismatch}")
    assert n_mismatch == 0, "_extinct mirror NOT exact"
    print("A2 PASS")


if __name__ == "__main__":
    main()
