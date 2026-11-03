"""record_s2_traces: record all eval_s2 games with record_trace.py,
verify each re-run's final cash against eval_s2_results.jsonl
(exact 0.00 required), and append the S2 section to traces/INDEX.md.

Same reproducibility protocol as record_v53_traces.py (shared seeding
random.seed(7919*seed + 104729*policy_side + 31); the S2 files add no
global-random consumption, so re-runs reproduce bit-for-bit).
record_trace.load_agent_fn is monkeypatched to eval_v53's loader rule.
"""

import importlib.util
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import eval_s2
import eval_v53
import record_trace

HERE = os.path.dirname(os.path.abspath(__file__))


def _patched_load(path, kind):
    return eval_v53.load_agent_fn(path)


record_trace.load_agent_fn = _patched_load

_orig_record_game = record_trace.record_game


def _seeded_record_game(policy_file, opp_kind, seed, policy_side, tag=None):
    random.seed(eval_v53.rng_seed(seed, policy_side))
    return _orig_record_game(policy_file, opp_kind, seed, policy_side, tag)


record_trace.record_game = _seeded_record_game

JOBS = []
for phase in ("selfcheck", "posture", "control", "abl_tree", "abl_harv"):
    for (tag, a_file, b_file, seed, side) in eval_s2.jobs_for(phase):
        JOBS.append((a_file, os.path.splitext(b_file)[0], seed, side, tag))


def load_expected():
    exp = {}
    with open(os.path.join(HERE, "eval_s2_results.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            exp[(r["tag"], r["seed"], r["policy_side"])] = (
                r["my_cash"], r["opp_cash"])
    return exp


NAMES = {
    "base_v53": "base_v53（基座）",
    "switcher_harvest": "变速陪练·收割型（步100裁员）",
    "switcher_sprint": "变速陪练·冲刺型（步150裁员）",
    "switcher_treewar": "变速陪练·养树战型（步50–100加预算）",
    "fork_s2_posture": "fork_s2_posture（姿态全开）",
    "fork_s2_no_tree": "fork_s2_no_tree（消融：关反养树）",
    "fork_s2_no_harv": "fork_s2_no_harv（消融：关反收割）",
    "fork_s2_off": "fork_s2_off（消融：分类器关）",
}


def main():
    expected = load_expected()
    missing = [j for j in JOBS
               if (j[4], j[2], j[3]) not in expected]
    if missing:
        print("WARNING: jobs without eval results:", missing)
    jobs = [j for j in JOBS if (j[4], j[2], j[3]) in expected]
    results = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(record_trace._job, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            my = r["final_cash"][r["side"]]
            op = r["final_cash"][1 - r["side"]]
            emy, eop = expected[(r["tag"], r["seed"], r["side"])]
            r["my_cash"], r["opp_cash"] = my, op
            r["exp_my"], r["exp_opp"] = emy, eop
            r["diff_my"] = round(my - emy, 4)
            r["diff_opp"] = round(op - eop, 4)
            r["ok"] = (r["diff_my"] == 0.0 and r["diff_opp"] == 0.0)
            results.append(r)
            print(f"{r['tag']} seed{r['seed']} side{r['side']}: "
                  f"rerun {my:.2f}:{op:.2f} vs eval {emy:.2f}:{eop:.2f} "
                  f"diff {r['diff_my']:+.2f}/{r['diff_opp']:+.2f} "
                  f"{'OK' if r['ok'] else 'MISMATCH'}", flush=True)
    results.sort(key=lambda r: (r["tag"], r["seed"], r["side"]))
    bad = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(bad)}/{len(results)} games reproduce "
          f"the eval final scores exactly (0.00).")
    if bad:
        print("MISMATCHED:", [(r["tag"], r["seed"], r["side"])
                               for r in bad])

    lines = ["", "## S2 姿态切换层（eval_s2，2026-11-03）", "",
             "| 版本 | 对手 | seed | 先后手（我方） | 终分（我方 : 对手） "
             "| 原记录 | 核对差 | 文件 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        side_txt = "P0 先手" if r["side"] == 0 else "P1 后手"
        lines.append(
            f"| {NAMES.get(r['policy'], r['policy'])}（{r['tag']}） "
            f"| {NAMES.get(r['opp'], r['opp'])} "
            f"| {r['seed']} | {side_txt} "
            f"| {r['my_cash']:.2f} : {r['opp_cash']:.2f} "
            f"| {r['exp_my']:.2f} : {r['exp_opp']:.2f} "
            f"| {r['diff_my']:+.2f} / {r['diff_opp']:+.2f} "
            f"| [{r['file']}]({r['file']}) |")
    lines.append("")
    with open(os.path.join(HERE, "traces", "INDEX.md"), "a") as f:
        f.write("\n".join(lines))
    print("INDEX.md S2 section appended.")


if __name__ == "__main__":
    main()
