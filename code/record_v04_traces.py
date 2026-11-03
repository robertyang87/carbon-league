"""record_v04_traces: record all v0.4 eval games with record_trace.py,
verify each re-run's final cash against eval_v0_4_results.jsonl
(exact 0.00 required -- the policies are deterministic), and append
the v0.4 section to traces/INDEX.md.

Games: eval_v04 (v0.4 vs juzi, seeds 1000-1003 x2 sides),
eval_v04_ex (v0.4 vs example, seeds 2000-2001 x2 sides),
ablV04notree (v0.4 tree-warfare OFF vs juzi, seed 1001 x2 sides).
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import record_trace

HERE = os.path.dirname(os.path.abspath(__file__))

JOBS = []
for seed in (1000, 1001, 1002, 1003):
    for side in (0, 1):
        JOBS.append(("policy_v0_4.py", "reference_juzi", seed, side,
                     "eval_v04"))
for seed in (2000, 2001):
    for side in (0, 1):
        JOBS.append(("policy_v0_4.py", "example", seed, side,
                     "eval_v04_ex"))
for side in (0, 1):
    JOBS.append(("policy_v0_4_notree.py", "reference_juzi", 1001, side,
                 "ablV04notree"))


def load_expected():
    exp = {}
    with open(os.path.join(HERE, "eval_v0_4_results.jsonl")) as f:
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
    with ProcessPoolExecutor(max_workers=4) as ex:
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
            r["ok"] = r["diff_my"] == 0.0 and r["diff_opp"] == 0.0
            results.append(r)
            print(f"{r['tag']} seed{r['seed']} side{r['side']}: "
                  f"{my:.2f}:{op:.2f} diff {r['diff_my']:+.2f}/"
                  f"{r['diff_opp']:+.2f} {'OK' if r['ok'] else 'MISMATCH'}",
                  flush=True)
    order = {"eval_v04": 0, "eval_v04_ex": 1, "ablV04notree": 2}
    results.sort(key=lambda r: (order[r["tag"]], r["seed"], r["side"]))
    lines = ["", "## v0.4 补录（policy_v0_4 / policy_v0_4_notree，2026-11-03）",
             "",
             "| 版本 | 对手 | seed | 先后手（我方） | 终分（我方 : 对手） "
             "| 原记录 | 核对差 | 文件 |",
             "|---|---|---|---|---|---|---|---|"]
    opp_name = {"reference_juzi": "juzi（reference_juzi）",
                "example": "example"}
    for r in results:
        side_txt = "P0 先手" if r["side"] == 0 else "P1 后手"
        lines.append(
            f"| {r['policy']}（{r['tag']}） | {opp_name[r['opp']]} "
            f"| {r['seed']} | {side_txt} "
            f"| {r['my_cash']:.2f} : {r['opp_cash']:.2f} "
            f"| {r['exp_my']:.2f} : {r['exp_opp']:.2f} "
            f"| {r['diff_my']:+.2f} / {r['diff_opp']:+.2f} "
            f"| [{r['file']}]({r['file']}) |")
    lines.append("")
    with open(os.path.join(HERE, "traces", "INDEX.md"), "a") as f:
        f.write("\n".join(lines))
    bad = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(bad)}/{len(results)} traces match exactly.")
    if bad:
        print("MISMATCHED:", [(r["tag"], r["seed"], r["side"]) for r in bad])


if __name__ == "__main__":
    main()
