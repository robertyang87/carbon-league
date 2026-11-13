"""Regression fixtures for League pool semantics in rl2_ppo_train.

league.json persists across launches and the constructor merge only ever
adds: a member dropped from --pool (or SELF under --self-prior 0) used to
stay in the sampling pool forever, so receipts quoting the config pool did
not describe the sampling pool. These fixtures pin the fixed semantics:
- without reconcile(), the legacy merge is unchanged;
- reconcile() retires unwanted non-snapshot members (prior -> 0, record
  kept), which is what --pool-authoritative invokes from main();
- retired members are never sampled and stay retired across a relaunch.

Run: python test_league_pool.py  (from code/, or from an acceptance tree)
"""
import json
import os
import sys
import tempfile

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

RES = []


def check(name, ok):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name, flush=True)


def seed_table(path):
    tab = {
        "botA.py": {"prior": 0.5, "n": 10, "wr": 0.6, "margin": 100.0},
        "botB.py": {"prior": 0.3, "n": 4, "wr": 0.4, "margin": -50.0},
        "SNAP:cnet2:/tmp/snap_1.npz": {"prior": 0.5, "n": 2, "wr": 0.5, "margin": 0.0, "snap": True},
        "SELF": {"prior": 1.0, "n": 3, "wr": 0.5, "margin": 0.0},
    }
    json.dump(tab, open(path, "w"), indent=1)
    return tab


def main():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "league.json")
        seed_table(path)

        lg = PT.League(path, [["botA.py", 0.5]], snap_prior=0.5, pfsp=1.0)
        check("legacy merge keeps unlisted member active",
              lg.tab["botB.py"]["prior"] == 0.3 and lg.tab["botA.py"]["prior"] == 0.5)

        retired = lg.reconcile({"botA.py"}, keep_self=True)
        check("reconcile retires exactly the unlisted non-snapshot member",
              retired == ["botB.py"])
        check("retirement zeroes the prior but keeps the record",
              lg.tab["botB.py"]["prior"] == 0.0 and lg.tab["botB.py"]["n"] == 4
              and lg.tab["botB.py"]["wr"] == 0.4 and lg.tab["botB.py"]["margin"] == -50.0)
        check("reconcile leaves listed members and snapshots untouched",
              lg.tab["botA.py"]["prior"] == 0.5
              and lg.tab["SNAP:cnet2:/tmp/snap_1.npz"]["prior"] == 0.5
              and lg.tab["SELF"]["prior"] == 1.0)

        rng = np.random.default_rng(7)
        draws = {lg.sample(rng) for _ in range(500)}
        check("retired member is never sampled", "botB.py" not in draws and "botA.py" in draws)

        lg.save()
        lg2 = PT.League(path, [], snap_prior=0.5, pfsp=1.0)
        check("retirement persists across a relaunch merge",
              lg2.tab["botB.py"]["prior"] == 0.0 and "botB.py" in lg2.tab)

        seed_table(path)
        lg3 = PT.League(path, [["botA.py", 0.5]], snap_prior=0.5, pfsp=1.0)
        retired3 = lg3.reconcile({"botA.py"}, keep_self=False)
        check("SELF is retired when the launch does not want self-play",
              sorted(retired3) == ["SELF", "botB.py"] and lg3.tab["SELF"]["prior"] == 0.0)

        fresh = os.path.join(td, "fresh_league.json")
        lg4 = PT.League(fresh, [["botA.py", 0.5], ["botB.py", 0.3]], snap_prior=0.5, pfsp=1.0)
        check("fresh table reconciles to a no-op",
              lg4.reconcile({"botA.py", "botB.py"}, keep_self=False) == [])

    print(f"==== {sum(RES)}/{len(RES)} checks passed ====", flush=True)
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
