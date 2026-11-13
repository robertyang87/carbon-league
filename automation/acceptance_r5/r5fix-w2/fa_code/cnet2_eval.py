"""cnet2_eval -- paired evaluation summary (wraps evo.run_pairs).
Usage: python cnet2_eval.py A.py B.py SEED0 NSEEDS TAG
Prints n, wins, mean +- se, worst game, mean ms/step, worst step ms, ledger means.
Pauses the PPO run (pid in ../data/rl10/pid, plus children) during the eval when CNET2_PAUSE_RL=1.
"""
import json
import os
import signal
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "1"
os.environ.setdefault("EVO_WORKERS", "3")
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from evo import _play  # noqa: E402

LED = ("deposits", "tree_income", "death_cargo", "plant_count", "plant_spend", "seize_count", "recruit_count")


def run_pairs(jobs, tag="cnet2"):
    """like evo.run_pairs but keeps plant/seize/recruit ledger fields."""
    tasks = [(tag, a, b, s, side) for a, b, s0, n in jobs for s in range(s0, s0 + n) for side in (0, 1)]
    rows = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("EVO_WORKERS", "3"))) as ex:
        for r in ex.map(_play, tasks):
            slim = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
            if "ledger_pol" in r:
                for k in LED:
                    slim["pol_" + k] = r["ledger_pol"].get(k)
                    slim["opp_" + k] = r["ledger_opp"].get(k)
            rows.append(slim)
    return rows


def rl_pids():
    pids = []
    for f in ("../data/rl10/pid", "../data/rl2/run1/pid"):
        try:
            pid = int(open(os.path.join(HERE, f)).read().strip())
            os.kill(pid, 0)
        except Exception:
            continue
        out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
        pids += [pid] + [int(x) for x in out]
    return pids


def summ(rows):
    ok = [r for r in rows if "margin" in r]
    m = [r["margin"] for r in ok]
    se = st.stdev(m) / len(m) ** 0.5 if len(m) > 1 else 0
    w = min(ok, key=lambda r: r["margin"])
    led = {k: round(st.mean(r[k] for r in ok), 1) for k in
           ["pol_" + k for k in LED] + ["opp_" + k for k in LED]}
    return {"n": len(ok), "err": len(rows) - len(ok), "wins": sum(r["win"] for r in ok),
            "mean": round(st.mean(m), 1), "se": round(se, 1),
            "worst": {"seed": w["seed"], "side": w["policy_side"], "margin": w["margin"]},
            "a_ms_mean": round(st.mean(r["a_ms_per_step"] for r in ok), 1),
            "a_worst_ms": max(r["a_worst_step_ms"] for r in ok), "ledger": led}


def main(a, b, s0, n, tag):
    pids = rl_pids() if os.environ.get("CNET2_PAUSE_RL") == "1" else []
    for p in pids:
        try:
            os.kill(p, signal.SIGSTOP)
        except Exception:
            pass
    try:
        rows = run_pairs([(a, b, int(s0), int(n))], tag=tag)
    finally:
        for p in pids:
            try:
                os.kill(p, signal.SIGCONT)
            except Exception:
                pass
    errs = [r.get("error") for r in rows if "error" in r]
    s = summ(rows)
    s.update(tag=tag, a=a, b=b, seed0=int(s0))
    if errs:
        s["first_error"] = errs[0]
    print(json.dumps(s))
    with open(os.path.join(HERE, "../data/cnet2_eval_results.jsonl"), "a") as f:
        f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:6])
