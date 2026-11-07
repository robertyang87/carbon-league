"""gate -- promotion gate: candidate NN weights (with the v9 rule layer) vs v9.

  python3 gate.py WEIGHTS.npz TAG [N_SEEDS=32] [SEED0=900000]

Plays N_SEEDS paired seeds x both sides (2*N games) of exp/cand_<TAG>.py vs
exp/submission_nn_v9.py and appends a summary row to ../data/gate_results.jsonl.
PASS criterion (auto-submit threshold agreed with the owner, 2026-11-07):
  n >= 64, win rate >= 0.65, mean >= +1000, mean - 2*se > 0,
  worst game > -5000 (no collapse);  OR  n >= 30 and win rate > 0.80.
  Always: no errors, worst step < 500 ms.
Seeds 900000+ are disjoint from RL training seeds (500000+).
"""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evo import run_pairs  # noqa: E402

V9 = "exp/submission_nn_v9.py"
WRAP = '''import sys; sys.path.insert(0, {here!r})
import nn_agent as _A
_A.ENDGAME_BRAVE = False
_A.STALL_OFF_LEFT = 30
_W = _A.load_weights({w!r})
_AG = None
def agent(o, c):
    global _AG
    if _AG is None or o['step'] == 0:
        _AG = _A.make_agent(_W)
    return _AG(o, c)
'''


def make_wrapper(weights, tag):
    path = os.path.join(HERE, "exp", f"cand_{tag}.py")
    with open(path, "w") as f:
        f.write(WRAP.format(here=HERE, w=os.path.abspath(weights)))
    return os.path.relpath(path, HERE)


def gate(weights, tag, n_seeds=32, seed0=900000, opp=V9):
    cand = make_wrapper(weights, tag)
    rows = run_pairs([(cand, opp, seed0, n_seeds)], tag=f"gate_{tag}", log=False)
    m = [r["margin"] for r in rows if "margin" in r]
    err = sum(1 for r in rows if "error" in r)
    n = len(m)
    mean = st.mean(m) if m else 0.0
    se = st.stdev(m) / n ** 0.5 if n > 1 else float("inf")
    wr = sum(1 for x in m if x > 0) / max(1, n)
    worst = min(m) if m else None
    wms = max((r.get("a_worst_step_ms", 0) for r in rows if "margin" in r), default=0)
    big = (n >= 64 and wr >= 0.65 and mean >= 1000 and mean - 2 * se > 0 and worst > -5000)
    alt = (n >= 30 and wr > 0.80)          # owner's second rule (2026-11-07): >80% wins over >=30 games
    ok = err == 0 and wms < 500 and (big or alt)
    rec = {"tag": tag, "weights": os.path.abspath(weights), "opp": opp, "seed0": seed0, "n": n, "err": err,
           "wins": sum(1 for x in m if x > 0), "winrate": round(wr, 3), "mean": round(mean, 1),
           "se": round(se, 1), "worst": round(worst, 1) if worst is not None else None,
           "worst_ms": wms, "pass": ok,
           "ledger": {k: round(st.mean([r.get(k) or 0 for r in rows if "margin" in r]), 1)
                      for k in ("pol_deposits", "pol_tree_income", "pol_death_cargo",
                                "opp_deposits", "opp_tree_income", "opp_death_cargo")}}
    with open(os.path.join(HERE, "..", "data", "gate_results.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    return rec


if __name__ == "__main__":
    gate(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 32,
         int(sys.argv[4]) if len(sys.argv) > 4 else 900000)
