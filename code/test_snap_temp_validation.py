"""Regression fixture for RL2_SNAP_TEMP code-level validation.

The snapshot evaluation temperature used to be read raw from the
environment at two rollout sites: a typo (5, -1, NaN) silently changed
what an evaluation measured (this is the operational failure behind
the offline-eval env lesson). rl2_ppo_rollout now validates it in
_snap_temp() -- [0, 2], 0 = greedy -- and fails fast at import.

Run: python test_snap_temp_validation.py  (from code/, or an acceptance tree)
"""
import os
import subprocess
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LEARNER_THREADS"):
    os.environ[name] = "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_HERE, "rl2_ppo_rollout.py")):
    CODE = _HERE
else:
    CODE = os.path.abspath(os.path.join(_HERE, "../../code"))

RES = []


def check(name, ok):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name, flush=True)


DRIVER = """
import sys
sys.path.insert(0, sys.argv[1])
import rl2_ppo_rollout as RO
print("SNAP_TEMP:", RO._snap_temp())
"""


def probe(value):
    env = dict(os.environ)
    if value is None:
        env.pop("RL2_SNAP_TEMP", None)
    else:
        env["RL2_SNAP_TEMP"] = value
    return subprocess.run([sys.executable, "-c", DRIVER, CODE],
                          capture_output=True, text=True, timeout=300, env=env)


def main():
    cp = probe(None)
    check("unset defaults to greedy 0", cp.returncode == 0 and "SNAP_TEMP: 0.0" in cp.stdout)
    cp = probe("0.5")
    check("operational 0.5 accepted", cp.returncode == 0 and "SNAP_TEMP: 0.5" in cp.stdout)
    for bad in ("5", "-1", "nan"):
        cp = probe(bad)
        check(f"invalid value {bad!r} refused at import",
              cp.returncode != 0 and "RL2_SNAP_TEMP" in cp.stderr)
    cp = probe("abc")
    check("non-numeric value refused", cp.returncode != 0)

    print(f"==== {sum(RES)}/{len(RES)} checks passed ====", flush=True)
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
