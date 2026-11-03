"""evo -- self-play evolution harness for the v53/no-camp rule line.

Variants = a parent file + PARAMS overrides appended at the end
(Bot() copies PARAMS at step 0, so a trailing PARAMS.update() is exact).
Evaluation reuses eval_v53.play_game verbatim (FastCarbon, canonical
seeding, ledger), paired: every seed is played from both sides.

Usage (library):  from evo import make_variant, run_pairs
CLI:  python3 evo.py match A.py B.py SEED0 N [tag]
"""

import json
import os
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EXP = os.path.join(HERE, "exp")
RESULTS = os.path.join(HERE, "..", "data", "evo_results.jsonl")
WORKERS = int(os.environ.get("EVO_WORKERS", "4"))


def make_variant(parent, overrides, name):
    """Write exp/<name>.py = parent + PARAMS.update(overrides). Returns rel path."""
    os.makedirs(EXP, exist_ok=True)
    src = open(os.path.join(HERE, parent)).read()
    src += "\n\n# --- evo overrides ---\nPARAMS.update(%r)\n" % (overrides,)
    path = os.path.join(EXP, name + ".py")
    with open(path, "w") as f:
        f.write(src)
    return os.path.relpath(path, HERE)


def _play(task):
    import eval_v53
    try:
        return eval_v53.play_game(task)
    except Exception as e:  # a crashing variant must not kill the batch
        tag, a, b, seed, side = task
        return {"tag": tag, "a_file": a, "b_file": b, "seed": seed,
                "policy_side": side, "error": repr(e)[:300]}


def run_pairs(jobs, tag="evo", log=True):
    """jobs: list of (a_file, b_file, seed0, n). Returns list of rows."""
    tasks = [(tag, a, b, s, side) for a, b, s0, n in jobs
             for s in range(s0, s0 + n) for side in (0, 1)]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for r in ex.map(_play, tasks):
            slim = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
            if "ledger_pol" in r:
                for k in ("deposits", "tree_income", "death_cargo"):
                    slim["pol_" + k] = r["ledger_pol"].get(k)
                    slim["opp_" + k] = r["ledger_opp"].get(k)
            slim["t"] = time.time()
            rows.append(slim)
            if log:
                with open(RESULTS, "a") as f:
                    f.write(json.dumps(slim) + "\n")
    return rows


def summarize(rows):
    m = [r["margin"] for r in rows if "margin" in r]
    err = sum(1 for r in rows if "error" in r)
    if not m:
        return {"n": 0, "err": err}
    se = st.stdev(m) / len(m) ** 0.5 if len(m) > 1 else float("nan")
    return {"n": len(m), "wins": sum(1 for r in rows if r.get("win")),
            "mean": round(st.mean(m), 1), "se": round(se, 1), "err": err,
            "worst_ms": max(r.get("a_worst_step_ms", 0) for r in rows if "margin" in r)}


if __name__ == "__main__":
    if sys.argv[1] == "match":
        a, b, s0, n = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
        tag = sys.argv[6] if len(sys.argv) > 6 else "match"
        print(json.dumps(summarize(run_pairs([(a, b, s0, n)], tag))))
