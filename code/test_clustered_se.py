"""Regression fixtures for seed-clustered SE in the PPO eval summaries.

Both rl2_ppo_train.summarize_eval and rl2_ppo_eval.summarize used to
compute the SE per game (iid), treating the two seats of one seed as
independent and understating the SE by up to sqrt(2) (section 20 R2:
the independent unit is the seed). The SE now runs over per-seed
cluster means, and rows carry n_seeds alongside n.

Emission caliber (Codex re-review item 3): with a single seed the
clustered SE is inestimable and the summary rows must emit se/se_u as
null -- never 0.0, which falsely claims a zero-width interval on data
that visibly disperses (the helper returns None; the emission layer
must not convert it). Mirrors stage1_teacher_gate's None for n < 2.

Run: python test_clustered_se.py  (from code/, or an acceptance tree)
"""
import os
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LEARNER_THREADS"):
    os.environ[name] = "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_HERE, "rl2_ppo_train.py")):
    CODE = _HERE
else:
    CODE = os.path.abspath(os.path.join(_HERE, "../../code"))
sys.path.insert(0, CODE)

import numpy as np
import rl2_ppo_train as PT
import rl2_ppo_eval as EV

RES = []


def check(name, ok):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name, flush=True)


def main():
    # seed cluster means are [10, 20, 30, 40]: clustered se =
    # std([10,20,30,40], ddof=1) / sqrt(4) = 6.4549...
    pairs = [(1, 0.0), (1, 20.0), (2, 10.0), (2, 30.0),
             (3, 20.0), (3, 40.0), (4, 30.0), (4, 50.0)]
    se, n_cl = PT._clustered_se(pairs)
    expect = float(np.std([10, 20, 30, 40], ddof=1) / 2.0)
    check("clustered SE matches the hand-computed value",
          n_cl == 4 and abs(se - expect) < 1e-9)
    iid = float(np.std([v for _, v in pairs], ddof=1) / np.sqrt(8))
    check("clustered SE exceeds the iid figure on paired data", se > iid)

    results = [{"games": [{"opp": "X", "seed": s, "margin": v} for s, v in pairs]}]
    row = PT.summarize_eval(85, results)[0]
    check("trainer summary carries clustered se + n_seeds",
          row["n"] == 8 and row["n_seeds"] == 4 and abs(row["se"] - round(expect, 1)) < 1e-9)

    srow = EV.summarize({"X": pairs})["X"]
    check("standalone eval summary agrees (se clustered, se_iid kept)",
          srow["n"] == 8 and srow["n_seeds"] == 4
          and abs(srow["se"] - round(expect, 1)) < 1e-9
          and abs(srow["se_iid"] - round(iid, 1)) < 1e-9)

    se1, n1 = PT._clustered_se([(7, 5.0), (7, -5.0)])
    check("a single seed cluster yields no SE (None), never an iid fallback",
          se1 is None and n1 == 1)

    # Emission layer (item 3): a single seed's summary row must carry
    # se/se_u as None (JSON null = inestimable), not 0.0 -- even though
    # the two seats visibly disperse (iid se of [100, -50] is ~75).
    import json as _json
    res1 = [{"games": [{"opp": "X", "seed": 7, "margin": 100.0},
                       {"opp": "X", "seed": 7, "margin": -50.0}]}]
    row1 = PT.summarize_eval(85, res1)[0]
    check("trainer emission: single-seed se and se_u are None, never 0.0",
          row1["n_seeds"] == 1 and row1["se"] is None and row1["se_u"] is None
          and '"se": null' in _json.dumps(row1))
    srow1 = EV.summarize({"X": [(7, 100.0), (7, -50.0)]})["X"]
    check("standalone emission: single-seed se is None while se_iid shows dispersion",
          srow1["n_seeds"] == 1 and srow1["se"] is None and srow1["se_iid"] > 0.0)

    print(f"==== {sum(RES)}/{len(RES)} checks passed ====", flush=True)
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
