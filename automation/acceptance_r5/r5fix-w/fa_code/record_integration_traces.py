"""record_integration_traces: record every eval_integration game with
record_trace.py, verify each re-run's final cash against
eval_integration_results.jsonl (exact 0.00 required), and append the
integration section to traces/INDEX.md.

Reproducibility protocol identical to eval_v53/record_v53_traces:
random.seed(7919*seed + 104729*policy_side + 31) before the game,
loader rule policy_* -> make_agent() else module agent().

Usage: python3 record_integration_traces.py [batch ...]
(default: all batches present in the results file)
"""

import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import eval_v53
import record_trace

HERE = os.path.dirname(os.path.abspath(__file__))

_orig_load = record_trace.load_agent_fn


def _patched_load(path, kind):
    return eval_v53.load_agent_fn(path)


record_trace.load_agent_fn = _patched_load

_orig_record_game = record_trace.record_game


def _seeded_record_game(policy_file, opp_kind, seed, policy_side, tag=None):
    random.seed(eval_v53.rng_seed(seed, policy_side))
    return _orig_record_game(policy_file, opp_kind, seed, policy_side, tag)


record_trace.record_game = _seeded_record_game


def load_expected(batches):
    exp = {}
    jobs = []
    with open(os.path.join(HERE, "eval_integration_results.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if batches and r["batch"] not in batches:
                continue
            key = (r["batch"], r["seed"], r["policy_side"])
            exp[key] = (r["my_cash"], r["opp_cash"])
            jobs.append((r["a_file"], os.path.splitext(r["b_file"])[0],
                         r["seed"], r["policy_side"], r["batch"]))
    return exp, jobs


def main():
    batches = sys.argv[1:] or None
    expected, jobs = load_expected(batches)
    results = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(record_trace._job, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            key = (r["tag"], r["seed"], r["side"])
            my = r["final_cash"][r["side"]]
            op = r["final_cash"][1 - r["side"]]
            emy, eop = expected[key]
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

    lines = ["", "## 集成线 fork-v0.2（eval_integration，2026-11-03）", "",
             "| 批次 | 版本（A） | 对手（B） | seed | 先后手（A） "
             "| 终分（A : B） | 核对差 | 文件 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        side_txt = "P0 先手" if r["side"] == 0 else "P1 后手"
        lines.append(
            f"| {r['tag']} | {r['policy']} | {r['opp']} | {r['seed']} "
            f"| {side_txt} | {r['my_cash']:.2f} : {r['opp_cash']:.2f} "
            f"| {r['diff_my']:+.2f} / {r['diff_opp']:+.2f} "
            f"| [{r['file']}]({r['file']}) |")
    lines.append("")
    with open(os.path.join(HERE, "traces", "INDEX.md"), "a") as f:
        f.write("\n".join(lines))
    print("INDEX.md integration section appended.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
