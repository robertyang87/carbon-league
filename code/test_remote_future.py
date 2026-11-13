"""Regression fixtures for split-mode remote file selection.

_Remote used to consume any trajectory file not older than
it - max_stale, including files stamped with a behaviour iteration
AHEAD of the learner (e.g. a previous run's actor sharing the remote
directory): future trajectories silently entered training, newest
first. _select_remote_files now enforces the full staleness contract
[cur_it - max_stale, cur_it]: future files are neither consumed nor
deleted (they become selectable if the learner reaches their
iteration), stale files are handed back for deletion, and the keep
order stays newest-first.

Run: python test_remote_future.py  (from code/, or an acceptance tree)
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

import rl2_ppo_train as PT

RES = []


def check(name, ok):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name, flush=True)


def main():
    names = ["it000089_10.pkl.z", "it000090_10.pkl.z", "it000095_10.pkl.z",
             "it000100_10.pkl.z", "it000101_10.pkl.z", "it000250_10.pkl.z"]
    keep, stale, future = PT._select_remote_files(names, cur_it=100, max_stale=10)
    check("stale boundary is exclusive below, inclusive at the floor",
          stale == ["it000089_10.pkl.z"] and "it000090_10.pkl.z" in keep)
    check("current iteration is consumable", "it000100_10.pkl.z" in keep)
    check("future files are neither kept nor stale",
          future == ["it000101_10.pkl.z", "it000250_10.pkl.z"]
          and not any(f in keep or f in stale for f in future))
    check("keep order is newest-first",
          keep == ["it000100_10.pkl.z", "it000095_10.pkl.z", "it000090_10.pkl.z"])

    keep, stale, future = PT._select_remote_files([], cur_it=100, max_stale=10)
    check("empty directory selects nothing", keep == stale == future == [])

    keep, stale, future = PT._select_remote_files(
        ["it000100_5.pkl.z"], cur_it=100, max_stale=0)
    check("zero staleness window keeps only the current iteration",
          keep == ["it000100_5.pkl.z"] and stale == [] and future == [])

    try:
        PT._select_remote_files(["garbage.pkl.z"], cur_it=100, max_stale=10)
        raised = False
    except ValueError:
        raised = True
    check("malformed names keep the historical loud failure", raised)

    print(f"==== {sum(RES)}/{len(RES)} checks passed ====", flush=True)
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
