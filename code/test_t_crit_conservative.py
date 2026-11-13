"""Regression fixture for the gate's t-critical fallback table.

Without scipy, _t_crit used to look UP to the next tabulated df (a
SMALLER critical value -- anti-conservative for in-between df, e.g.
df=11 got 2.179 against a true 2.201) and returned 1.96 beyond df=60
(the t tail is still ~1.98-2.00 there). The fallback table is now dense
for df 1..30 and rounds DOWN in df beyond it, so it never under-covers;
it stays labelled inexact (verdict-grade intervals want scipy or the
bootstrap regardless).

Run: python test_t_crit_conservative.py  (from code/, or an acceptance tree)
"""
import os
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LEARNER_THREADS"):
    os.environ[name] = "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_HERE, "stage1_teacher_gate.py")):
    CODE = _HERE
else:
    CODE = os.path.abspath(os.path.join(_HERE, "../../code"))
sys.path.insert(0, CODE)

from scipy import stats as _st
import stage1_teacher_gate as G

RES = []


def check(name, ok):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name, flush=True)


def main():
    true = {df: float(_st.t.ppf(0.975, df)) for df in range(1, 301)}
    # Force the stdlib fallback: a None module entry makes
    # "from scipy import stats" raise ImportError inside _t_crit.
    saved = sys.modules.get("scipy")
    sys.modules["scipy"] = None
    try:
        fb = {}
        flags = {}
        for df in range(1, 301):
            fb[df], flags[df] = G._t_crit(df)
    finally:
        if saved is not None:
            sys.modules["scipy"] = saved
        else:
            sys.modules.pop("scipy", None)
    check("fallback never under-covers (df 1..300, 3-decimal table slack)",
          all(fb[df] >= true[df] - 0.0015 for df in true))
    check("fallback is not absurdly conservative (within +0.35)",
          all(fb[df] <= true[df] + 0.35 for df in true))
    check("fallback stays labelled inexact", all(f is False for f in flags.values()))
    check("the old anti-conservative case df=11 is fixed", fb[11] >= 2.201 - 0.0015)
    check("far tail uses the df=120 value, not 1.96", fb[300] >= 1.98 - 0.0015)

    val, exact = G._t_crit(11)
    check("scipy path still exact when available", exact and abs(val - true[11]) < 1e-9)

    print(f"==== {sum(RES)}/{len(RES)} checks passed ====", flush=True)
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
