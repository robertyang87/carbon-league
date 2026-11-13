"""record_v05_traces: record all v0.5 eval games with record_trace.py,
verify each re-run's final cash against eval_v0_5_results.jsonl
(exact 0.00 required -- the policies are deterministic), and append
the v0.5 section to traces/INDEX.md.

Games: eval_v05 (v0.5 shipped, switch OFF, vs juzi seeds 1000-1003 x2),
eval_v05_ex (v0.5 vs example, seeds 2000-2001 x2),
v05sw (policy_v0_5_switch, switch ON ablation arm, vs juzi 1001-1002 x2).
The v05nosw ablation arm is byte-identical in logic to the shipped
policy_v0_5.py (only the flag comment differs), so its games are the
eval_v05 seed 1001/1002 games -- recorded once, under eval_v05.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import record_trace

HERE = os.path.dirname(os.path.abspath(__file__))

JOBS = []
for seed in (1000, 1001, 1002, 1003):
    for side in (0, 1):
        JOBS.append(("policy_v0_5.py", "reference_juzi", seed, side,
                     "eval_v05"))
for seed in (2000, 2001):
    for side in (0, 1):
        JOBS.append(("policy_v0_5.py", "example", seed, side,
                     "eval_v05_ex"))
for seed in (1001, 1002):
    for side in (0, 1):
        JOBS.append(("policy_v0_5_switch.py", "reference_juzi", seed, side,
                     "v05sw"))


def load_expected():
    exp = {}
    with open(os.path.join(HERE, "eval_v0_5_results.jsonl")) as f:
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
    order = {"eval_v05": 0, "eval_v05_ex": 1, "v05sw": 2}
    results.sort(key=lambda r: (order[r["tag"]], r["seed"], r["side"]))
    lines = ["", "## v0.5（policy_v0_5 / policy_v0_5_switch，2026-11-03）",
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
