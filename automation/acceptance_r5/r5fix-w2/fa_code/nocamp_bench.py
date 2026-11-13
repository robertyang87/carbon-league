#!/usr/bin/env python3
"""nocamp_bench -- verify the OpenAI-side no-camp claims under our protocol.

Groups (candidate = first named), seeds 1000-1003 x both sides = 8 games
each, canonical seeding random.seed(7919*seed + 104729*side + 31) with
policy = candidate (inside caseb_rl_probe.play), FastCarbon, full traces
in traces/, rows appended to traces/INDEX.md under the nocamp_bench
section.  One game per group is re-run in-process afterwards and must
match the recorded final cash exactly.

  1. nocamp_recon            vs base_v53   (validation group)
  2. policy_aux_seg2_zero    vs nocamp_recon (archived champion)
  3. fork_v0_1               vs nocamp_recon (only submittable face)

Results: nocamp_bench_results.json (rows + summaries).
"""

import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import caseb_rl_probe as probe  # noqa: E402
import record_trace  # noqa: E402

TAG = "nocamp_bench"
INDEX = os.path.join(record_trace.TRACES_DIR, "INDEX.md")
SECTION = "## no-camp 基准核对（nocamp_bench，2026-11-04）"
RESULTS = os.path.join(HERE, "nocamp_bench_results.json")

GROUPS = [
    ("nocamp_vs_base", "nocamp_recon.py", "base_v53.py"),
    ("aux_vs_nocamp", "caseb_rl_ckpt/policy_aux_seg2_zero.py",
     "nocamp_recon.py"),
    ("fork_vs_nocamp", "fork_v0_1.py", "nocamp_recon.py"),
]
SEEDS = [1000, 1001, 1002, 1003]


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


def append_index(all_rows, verify_map):
    exists = os.path.exists(INDEX)
    text = open(INDEX).read() if exists else ""
    lines = []
    if SECTION not in text:
        lines += ["", SECTION, "",
                  "| 标签 | 我方 | 对手 | seed | 先后手（我方） | "
                  "终分（我方 : 对手） | 核对差 | 文件 |",
                  "|---|---|---|---|---|---|---|---|"]
    for r in all_rows:
        side_txt = "P0 先手" if r["side"] == 0 else "P1 后手"
        chk = verify_map.get(r["file"], "未抽查")
        lines.append(
            f"| {r['tag']} | {r['policy']} | {r['opp']} | {r['seed']} "
            f"| {side_txt} | {r['my_cash']:.2f} : {r['opp_cash']:.2f} "
            f"| {chk} | [{r['file']}]({r['file']}) |")
    with open(INDEX, "a") as f:
        f.write("\n".join(lines) + "\n")


def main():
    jobs = []
    for gname, pol, opp in GROUPS:
        for seed in SEEDS:
            for side in (0, 1):
                jobs.append((gname, pol, opp, seed, side))
    print(f"nocamp_bench jobs={len(jobs)}", flush=True)
    results = {g: [] for g, _, _ in GROUPS}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {}
        for gname, pol, opp, seed, side in jobs:
            fut = ex.submit(probe._job, (pol, opp, seed, side, 0, TAG))
            futs[fut] = gname
        for fut in as_completed(futs):
            r = fut.result()
            gname = futs[fut]
            if not r["ok"]:
                print(f"FAIL {gname}: {r.get('error')}", flush=True)
                return 2
            r["group"] = gname
            results[gname].append(r)
            print(f"OK {gname} seed{r['seed']} side{r['side']} "
                  f"margin={r['margin']:.0f} ({r['sec']}s)", flush=True)

    # ---- rerun verification: first game of each group, in-process ----
    verify_map = {}
    bad = 0
    for gname, pol, opp in GROUPS:
        r0 = sorted(results[gname],
                    key=lambda r: (r["seed"], r["side"]))[0]
        rr = probe.play((pol, opp, r0["seed"], r0["side"], 0, TAG))
        match = rr["final_cash"] == r0["final_cash"]
        verify_map[r0["file"]] = "+0.00 / +0.00" if match else "MISMATCH"
        bad += 0 if match else 1
        print(f"VERIFY {gname} {r0['file']}: "
              f"{'MATCH' if match else 'MISMATCH ' + str(rr['final_cash'])}",
              flush=True)

    all_rows = []
    summaries = {}
    for gname, _, _ in GROUPS:
        rows = sorted(results[gname], key=lambda r: (r["seed"], r["side"]))
        summaries[gname] = summarize(rows)
        all_rows += rows
        s = summaries[gname]
        print(f"SUMMARY {gname}: {s['wins']}W/{s['losses']}L "
              f"mean={s['margin_mean']:.1f} std={s['margin_std_sample']:.1f}",
              flush=True)
    append_index(all_rows, verify_map)
    with open(RESULTS, "w") as f:
        json.dump({"tag": TAG, "groups": summaries, "rows": all_rows},
                  f, ensure_ascii=False, indent=1)
    print(f"wrote {RESULTS}; index appended; verify_bad={bad}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
