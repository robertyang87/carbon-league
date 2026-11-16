"""Regression fixture for the handicap ("pressure") opponent spec in rl2_ppo_rollout.

"<spec>@C<amount>" plays the same opponent with a starting-cash head start. Pinned here:
- split_handicap parses only a trailing "@C<number>", leaves other specs (including paths
  that merely contain "@C") alone, and refuses SELF / zero / empty-base handicaps;
- a spec without the suffix plays exactly as before (same margin as the pre-change path:
  handicap 0, prev baseline 0);
- with the suffix the opponent really starts ahead by the amount, the returned record
  keeps the FULL spec as "opp" (the league key) and carries "handicap";
- the recorded per-step margin differences still sum to (final margin + handicap), i.e.
  the head start is not booked as a first-step loss, while "margin" is the true margin;
- the trainer's League accepts the variant as its own member with its own statistics.

Run: python test_rollout_handicap.py  (from code/, or an acceptance tree)
"""
import os
import sys
import tempfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LEARNER_THREADS"):
    os.environ[name] = "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_HERE, "rl2_ppo_rollout.py")):
    CODE = _HERE
else:
    CODE = os.path.abspath(os.path.join(_HERE, "../../code"))
sys.path.insert(0, CODE)

import numpy as np
import torch
import rl2_ppo_adapters as AD
import rl2_ppo_rollout as RO
import rl2_ppo_train as PT
from cnet2_model import CNet2

RES = []


def check(name, ok):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name, flush=True)


def refused(spec):
    try:
        RO.split_handicap(spec)
    except ValueError:
        return True
    return False


def main():
    S = RO.split_handicap
    check("plain bot spec untouched", S("exp/opp_1cz_v5.py") == ("exp/opp_1cz_v5.py", 0.0))
    check("plain snap spec untouched", S("SNAP:cnet2:/w/a.npz#T2") == ("SNAP:cnet2:/w/a.npz#T2", 0.0))
    check("SELF untouched", S("SELF") == ("SELF", 0.0))
    check("snap with handicap", S("SNAP:cnet2:/w/a.npz#T2@C90") == ("SNAP:cnet2:/w/a.npz#T2", 90.0))
    check("decimal handicap", S("exp/bot.py@C37.5") == ("exp/bot.py", 37.5))
    check("path merely containing @C is not a handicap",
          S("SNAP:cnet2:/data@Cache/a.npz") == ("SNAP:cnet2:/data@Cache/a.npz", 0.0))
    check("SELF@C refused", refused("SELF@C50"))
    check("zero handicap refused", refused("exp/bot.py@C0"))
    check("empty base refused", refused("@C50"))

    torch.manual_seed(0)
    torch.set_num_threads(1)
    ad = AD.get_adapter("cnet2")
    with tempfile.TemporaryDirectory() as td:
        learner = os.path.join(td, "learner.npz")
        opp = os.path.join(td, "opp.npz")
        AD.save_npz(ad.export(CNet2(C=16, NB=1, D=16, NH=2, L=1, E=8, G=4)), learner, {})
        AD.save_npz(ad.export(CNet2(C=16, NB=1, D=16, NH=2, L=1, E=8, G=4)), opp, {})
        base = f"SNAP:cnet2:{opp}"

        def play(spec, side):
            r = RO.play_batch({"adapter": "cnet2", "wkey": learner, "games": [(spec, 4242, side)],
                               "temp": 1.0, "greedy": True, "record": True, "rng_seed": 7})
            return r["games"]

        for side in (0, 1):
            g0 = play(base, side)[0]
            g1 = play(base + "@C90", side)[0]
            check(f"side {side}: plain game reports handicap 0 and its own spec",
                  g0["handicap"] == 0.0 and g0["opp"] == base)
            check(f"side {side}: plain game per-step margins sum to the margin",
                  abs(float(g0["dmargin"].sum()) - g0["margin"]) < 0.05)
            check(f"side {side}: variant keeps the full spec and carries the handicap",
                  g1["opp"] == base + "@C90" and g1["handicap"] == 90.0)
            check(f"side {side}: variant per-step margins sum to margin + handicap",
                  abs(float(g1["dmargin"].sum()) - (g1["margin"] + 90.0)) < 0.05)
            check(f"side {side}: the head start changes the game", g1["margin"] != g0["margin"])
            check(f"side {side}: record lengths unchanged in kind",
                  len(g1["recs"]) > 0 and len(g1["dmargin"]) == g1["steps"])
        # the head start is really on the opponent's side at step 0
        sim_me = play(base + "@C5000", 0)[0]
        check("a large head start shows up in the final margin (opponent side, not ours)",
              sim_me["margin"] < play(base, 0)[0]["margin"])
        self_games = play("SELF", 0)
        check("SELF still returns the mirrored pair",
              len(self_games) == 2 and self_games[0]["margin"] == -self_games[1]["margin"])

        lg = PT.League(os.path.join(td, "league.json"), [[base, 1.0], [base + "@C90", 0.5]], 0.5, 1.0)
        lg.update(base + "@C90", -300.0); lg.update(base, 200.0)
        check("League keeps the variant as its own member with its own record",
              lg.tab[base + "@C90"]["n"] == 1 and lg.tab[base]["n"] == 1
              and lg.tab[base + "@C90"]["wr"] == 0.0 and lg.tab[base]["wr"] == 1.0)
        rng = np.random.default_rng(0)
        drawn = {lg.sample(rng) for _ in range(200)}
        check("both members are sampled", drawn == {base, base + "@C90"})

    print(f"{sum(RES)}/{len(RES)} passed", flush=True)
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
