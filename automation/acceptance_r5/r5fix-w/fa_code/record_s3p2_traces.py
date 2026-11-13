"""record_s3p2_traces: record all eval_s3p2 games (main ablation +
sanity) with record_trace.py, verify each re-run's final cash against
eval_s3p2_results.jsonl (exact 0.00 required), and append the S3 v2 section
to traces/INDEX.md.

Same reproducibility protocol as record_v53_traces.py: seed
random.seed(7919*seed + 104729*policy_side + 31) before each game,
loader patched to eval_v53.load_agent_fn (fork_s3_* files do not start
with "policy_", so they load via the module-level agent()).
"""

import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import eval_s3p2 as eval_s3
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
for (tag, a_file, b_file, seed, side) in (eval_s3.jobs_for("main")
                                          + eval_s3.jobs_for("abl")
                                          + eval_s3.jobs_for("sanity")):
    JOBS.append((a_file, os.path.splitext(b_file)[0], seed, side, tag))


def load_expected():
    exp = {}
    with open(os.path.join(HERE, "eval_s3p2_results.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            exp[(r["tag"], r["seed"], r["policy_side"])] = (
                r["my_cash"], r["opp_cash"])
    return exp


def main():
    expected = load_expected()
    results = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(record_trace._job, j): j for j in JOBS}
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

    names = {
        "fork_s3_phase2": "fork-s3p2 全开",
        "fork_s3p2_honly": "fork-s3p2 仅猎杀",
        "fork_s3p2_off": "fork-s3p2 双关(sanity)",
        "fork_s3p2_norec": "fork-s3p2 相位+原补员闸(消融)",
        "base_v53": "base_v53（基座）",
    }
    lines = ["", "## S3 二版：终局相位重标定（eval_s3p2，2026-11-03）", "",
             "| 版本 | 对手 | seed | 先后手（我方） | 终分（我方 : 对手） "
             "| 原记录 | 核对差 | 文件 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        side_txt = "P0 先手" if r["side"] == 0 else "P1 后手"
        lines.append(
            f"| {names.get(r['policy'], r['policy'])}（{r['tag']}） "
            f"| {names.get(r['opp'], r['opp'])} "
            f"| {r['seed']} | {side_txt} "
            f"| {r['my_cash']:.2f} : {r['opp_cash']:.2f} "
            f"| {r['exp_my']:.2f} : {r['exp_opp']:.2f} "
            f"| {r['diff_my']:+.2f} / {r['diff_opp']:+.2f} "
            f"| [{r['file']}]({r['file']}) |")
    lines.append("")
    with open(os.path.join(HERE, "traces", "INDEX.md"), "a") as f:
        f.write("\n".join(lines))
    print("INDEX.md S3 v2 section appended.")


if __name__ == "__main__":
    main()
