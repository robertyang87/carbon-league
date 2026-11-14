"""Regression fixture for F6: the Terminal-U launch contract is a refusal.

u_mode_check() used to be advisory -- main() printed "[u-mode] WARNING"
and trained anyway, so a --reward u launch with the default gamma 0.995 /
rscale 1/5000 (terminal reward discounted to 0.22 and scaled by 2e-4)
ran with a critic fitting something that is not U. Pinned here:
- the canonical launch and every legacy (non-u) launch pass the gate;
- each contract violation is refused (SystemExit naming the problem);
- --allow-u-mode-problems lets the same launch through and returns the
  problems for the effective_params record;
- the real CLI refuses BEFORE creating the output directory.

Run: python test_ppo_u_mode_gate.py  (from code/, or an acceptance tree)
"""
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

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


def args(**kw):
    base = dict(reward="u", gamma=1.0, rscale=1.0, c_dep=0.0, c_lost=0.0, value_head="utility",
                allow_u_mode_problems=False)
    base.update(kw)
    return SimpleNamespace(**base)


def refused(a, needle):
    try:
        PT.u_mode_gate(a)
    except SystemExit as exc:
        return needle in str(exc) and "refusing to start" in str(exc)
    return False


def main():
    check("canonical Terminal-U launch passes", PT.u_mode_gate(args()) == [])
    check("legacy dense launch passes untouched",
          PT.u_mode_gate(args(reward="dense", value_head="tanh", gamma=0.995, rscale=2e-4, c_dep=0.1)) == [])
    check("gamma != 1 refused", refused(args(gamma=0.995), "gamma=0.995"))
    check("rscale != 1 refused", refused(args(rscale=1 / 5000.0), "rscale="))
    check("shaping terms refused", refused(args(c_dep=0.1), "c_dep/c_lost"))
    check("utility head without --reward u refused",
          refused(args(reward="dense", gamma=0.995, rscale=2e-4), "--value-head utility without --reward u"))
    p = PT.u_mode_gate(args(gamma=0.995, rscale=2e-4, allow_u_mode_problems=True))
    check("explicit override passes and returns the problems for the record", len(p) == 2)
    check("namespace without the flag (old callers) is refused, not crashed", refused(
        SimpleNamespace(reward="u", gamma=0.9, rscale=1.0, c_dep=0, c_lost=0, value_head="tanh"), "gamma=0.9"))

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "run")
        r = subprocess.run([sys.executable, os.path.join(CODE, "rl2_ppo_train.py"), "--init", "/nonexistent.npz",
                            "--out", out, "--reward", "u"], capture_output=True, text=True)
        check("CLI: default gamma/rscale with --reward u exits non-zero naming both",
              r.returncode != 0 and "gamma=0.995" in r.stderr and "rscale=" in r.stderr)
        check("CLI: refusal happens before the output directory is created", not os.path.exists(out))
        r = subprocess.run([sys.executable, os.path.join(CODE, "rl2_ppo_train.py"), "--init", "/nonexistent.npz",
                            "--out", out, "--reward", "u", "--allow-u-mode-problems"], capture_output=True, text=True)
        check("CLI: override gets past the gate (fails later, on the missing init)",
              r.returncode != 0 and "Terminal-U launch contract" not in r.stderr)

    print(f"{sum(RES)}/{len(RES)} passed", flush=True)
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
