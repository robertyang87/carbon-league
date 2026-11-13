"""record_v08_traces: record all v0.8-round eval games with
record_trace.py, verify each re-run's final cash against
eval_v0_8_results.jsonl (exact 0.00 required -- the policies are
deterministic), and append the v0.8 section to traces/INDEX.md.

Games: eval_v08_nosurv (policy_v0_7_nosurv vs juzi, seeds 1000-1003
x2), eval_v08_nosurv_ex (vs example, seeds 2000-2001 x2),
v08_replace (policy_v0_8_replace vs juzi, seeds 1001-1002 x2),
v08_dodge (policy_v0_8_dodge vs juzi, seeds 1001-1002 x2).
policy_v0_8.py itself is a byte copy of policy_v0_5.py (assembly
verdict), whose games are already recorded under policy_v0_5.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import record_trace

HERE = os.path.dirname(os.path.abspath(__file__))

JOBS = []
for seed in (1000, 1001, 1002, 1003):
    for side in (0, 1):
        JOBS.append(("policy_v0_7_nosurv.py", "reference_juzi", seed, side,
                     "eval_v08_nosurv"))
for seed in (2000, 2001):
    for side in (0, 1):
        JOBS.append(("policy_v0_7_nosurv.py", "example", seed, side,
                     "eval_v08_nosurv_ex"))
for seed in (1001, 1002):
    for side in (0, 1):
        JOBS.append(("policy_v0_8_replace.py", "reference_juzi", seed, side,
                     "v08_replace"))
        JOBS.append(("policy_v0_8_dodge.py", "reference_juzi", seed, side,
                     "v08_dodge"))


def load_expected():
    exp = {}
    with open(os.path.join(HERE, "eval_v0_8_results.jsonl")) as f:
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
    order = {"eval_v08_nosurv": 0, "eval_v08_nosurv_ex": 1,
             "v08_replace": 2, "v08_dodge": 3}
    results.sort(key=lambda r: (order[r["tag"]], r["seed"], r["side"]))
    lines = ["", "## v0.8（候选确认 + 拆分消融，2026-11-03）",
             "",
             "policy_v0_8.py 为 policy_v0_5.py 的字节级副本（组装裁决），"
             "其对局已在 policy_v0_5 名下录制，不重复登记。",
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
