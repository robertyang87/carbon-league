#!/usr/bin/env python3
"""nn_v1_check -- verify the successor-side claim "+409 vs no-camp"
for submission_nn_v1.py (fetched from GitHub main as nn_v1_from_main.py)
under our canonical protocol, identical to nocamp_bench:

  random.seed(7919*seed + 104729*side + 31) inside caseb_rl_probe.play,
  policy = candidate, seeds 1000-1003 x both sides = 8 games vs
  nocamp_recon.py, FastCarbon, full traces, INDEX section, first game
  re-run in-process and must match final cash exactly.

Loader note (disclosed, strategy body untouched): the stock bench loader
(caseb_rl_probe.load_agent_fresh) prefers mod.make_agent() called with no
arguments; in this packed submission that path dereferences an undefined
module global HERE and would load a sibling nn_bc_weights.npz, so the
stock loader raises NameError.  The submission's platform-contract entry
is its top-level agent(observation, configuration), which loads embedded
weights at import.  This check therefore patches ONLY the harness loader
(in each worker) to return mod.agent for this one file; every other
protocol element (seeding, side assignment, sim, ledger, traces) is
probe.play / record_trace unchanged.
"""

import importlib.util
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import caseb_rl_probe as probe  # noqa: E402
import record_trace  # noqa: E402

TAG = "nn_v1_check"
INDEX = os.path.join(record_trace.TRACES_DIR, "INDEX.md")
SECTION = "## NN v1 复核（nn_v1_check，2026-11-04）"
RESULTS = os.path.join(HERE, "nn_v1_check_results.json")
POL = "nn_v1_from_main.py"
OPP = "nocamp_recon.py"
GROUP = "nn_v1_vs_nocamp"
SEEDS = [1000, 1001, 1002, 1003]
CLAIM = 409.0

_orig_load = probe.load_agent_fresh


def _load_entry(path):
    if os.path.basename(path) == POL:
        name = "nnv1_entry_" + str(os.getpid())
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.agent
    return _orig_load(path)


def _patch():
    probe.load_agent_fresh = _load_entry


def job_fn(job):
    _patch()
    return probe._job(job)


def summarize(rows):
    ms = [r["margin"] for r in rows]
    wins = sum(1 for m in ms if m > 0)
    losses = sum(1 for m in ms if m < 0)
    draws = len(ms) - wins - losses

    def mean(key, led="ledger_pol"):
        return round(sum(r[led][key] for r in rows) / len(rows), 1)

    return {
        "n": len(rows),
        "wins": wins, "losses": losses, "draws": draws,
        "margin_mean": round(statistics.mean(ms), 1),
        "margin_std_sample": round(statistics.stdev(ms), 1) if len(ms) > 1
        else 0.0,
        "margin_min": round(min(ms), 1), "margin_max": round(max(ms), 1),
        "margins": [round(r["margin"], 1) for r in
                    sorted(rows, key=lambda r: (r["seed"], r["side"]))],
        "deposits_mean": mean("deposits"),
        "tree_income_mean": mean("tree_income"),
        "death_cargo_mean": mean("death_cargo"),
        "stranded_mean": mean("stranded"),
        "opp_deposits_mean": mean("deposits", "ledger_opp"),
        "opp_tree_income_mean": mean("tree_income", "ledger_opp"),
        "opp_death_cargo_mean": mean("death_cargo", "ledger_opp"),
        "opp_stranded_mean": mean("stranded", "ledger_opp"),
    }


def append_index(rows, verify_map):
    exists = os.path.exists(INDEX)
    text = open(INDEX).read() if exists else ""
    lines = []
    if SECTION not in text:
        lines += ["", SECTION, "",
                  "| 标签 | 我方 | 对手 | seed | 先后手（我方） | "
                  "终分（我方 : 对手） | 核对差 | 文件 |",
                  "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        side_txt = "P0 先手" if r["side"] == 0 else "P1 后手"
        chk = verify_map.get(r["file"], "未抽查")
        lines.append(
            f"| {r['tag']} | {r['policy']} | {r['opp']} | {r['seed']} "
            f"| {side_txt} | {r['my_cash']:.2f} : {r['opp_cash']:.2f} "
            f"| {chk} | [{r['file']}]({r['file']}) |")
    with open(INDEX, "a") as f:
        f.write("\n".join(lines) + "\n")


def main():
    jobs = [(POL, OPP, seed, side, 0, TAG)
            for seed in SEEDS for side in (0, 1)]
    print(f"nn_v1_check jobs={len(jobs)}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(job_fn, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            if not r["ok"]:
                print(f"FAIL: {r.get('error')}", flush=True)
                return 2
            r["group"] = GROUP
            rows.append(r)
            print(f"OK seed{r['seed']} side{r['side']} "
                  f"margin={r['margin']:.0f} ({r['sec']}s)", flush=True)

    rows.sort(key=lambda r: (r["seed"], r["side"]))
    # ---- rerun verification: first game, in-process ----
    _patch()
    r0 = rows[0]
    rr = probe.play((POL, OPP, r0["seed"], r0["side"], 0, TAG))
    match = rr["final_cash"] == r0["final_cash"]
    verify_map = {r0["file"]: "+0.00 / +0.00" if match else "MISMATCH"}
    print(f"VERIFY {r0['file']}: "
          f"{'MATCH' if match else 'MISMATCH ' + str(rr['final_cash'])}",
          flush=True)

    s = summarize(rows)
    lo, hi = CLAIM - 2 * s["margin_std_sample"], CLAIM + 2 * s["margin_std_sample"]
    reproduced = (lo <= s["margin_mean"] <= hi) and s["wins"] >= 5
    print(f"SUMMARY {GROUP}: {s['wins']}W/{s['losses']}L "
          f"mean={s['margin_mean']:.1f} std={s['margin_std_sample']:.1f} "
          f"band=[{lo:.1f},{hi:.1f}] reproduced={reproduced}", flush=True)
    append_index(rows, verify_map)
    with open(RESULTS, "w") as f:
        json.dump({"tag": TAG,
                   "claim": {"margin_mean": CLAIM, "wins": "6/8 (successor)"},
                   "verdict": {"reproduced": reproduced,
                               "band": [round(lo, 1), round(hi, 1)]},
                   "groups": {GROUP: s}, "rows": rows},
                  f, ensure_ascii=False, indent=1)
    print(f"wrote {RESULTS}; index appended; verify_match={match}",
          flush=True)
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
