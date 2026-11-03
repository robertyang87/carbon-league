"""record_rl2_traces: record all rl-v0.2 final eval games with
record_trace.py, verify each re-run's final cash against
eval_rl2_results.jsonl (exact 0.00 required -- rl_policy_v2's
sampling RNG is fixed-seeded per instance, so re-runs are
deterministic), and append the RL stage-2 section to traces/INDEX.md.

Games: rl2_v53 (vs base_v53, seeds 1000-1003 x2), rl2_shadow2
(vs shadow_alpha_v2, seeds 1000-1001 x2), rl2_v05 (vs policy_v0_5,
seed 1000 x2), rl2_example (seed 2000 x2). (No juzi games: retired
as a sparring opponent by user directive 2026-11-03 09:46.)

Run AFTER eval_rl2.py parts a (and c not needed here).
"""

import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import record_trace


def _seeded_job(job):
    """Same RNG protocol as eval_rl2 / eval_v53: seed the global
    random module per game (in the worker) before recording, so the
    re-run reproduces the evaluated game exactly (base_v53 draws
    from global random). Mirrors record_v53_traces.py's wrapper."""
    _pf, _opp, seed, side, _tag = job
    random.seed(7919 * seed + 104729 * side + 31)
    return record_trace._job(job)

HERE = os.path.dirname(os.path.abspath(__file__))

JOBS = []
for seed in (1000, 1001, 1002, 1003):
    for side in (0, 1):
        JOBS.append(("rl_policy_v2.py", "base_v53", seed, side,
                     "rl2_v53"))
for seed in (1000, 1001):
    for side in (0, 1):
        JOBS.append(("rl_policy_v2.py", "shadow_alpha_v2", seed, side,
                     "rl2_shadow2"))
for side in (0, 1):
    JOBS.append(("rl_policy_v2.py", "policy_v0_5", 1000, side,
                 "rl2_v05"))
    JOBS.append(("rl_policy_v2.py", "example", 2000, side,
                 "rl2_example"))


def load_expected():
    exp = {}
    with open(os.path.join(HERE, "eval_rl2_results.jsonl")) as f:
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
        futs = {ex.submit(_seeded_job, j): j for j in JOBS}
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
    order = {"rl2_v53": 0, "rl2_shadow2": 1, "rl2_v05": 2,
             "rl2_example": 3}
    results.sort(key=lambda r: (order[r["tag"]], r["seed"], r["side"]))
    lines = ["", "## RL 阶段二 rl-v0.2（信念通道 + 硬规则盾 PPO，2026-11-03）",
             "",
             "rl_policy_v2.py（权重 rl2_weights.npz）：阶段二 BC"
             "（信念碳图通道）热启动 + PPO 自博弈，探针集选点；"
             "评测见 eval_rl2_results.jsonl、eval_rl2_acceptance.jsonl"
             " 与 rl_stage2_report.md。",
             "",
             "| 版本 | 对手 | seed | 先后手（我方） | 终分（我方 : 对手） "
             "| 原记录 | 核对差 | 文件 |",
             "|---|---|---|---|---|---|---|---|"]
    opp_name = {"base_v53": "base_v53（claude-opus-v53 基座）",
                "shadow_alpha_v2": "shadow_alpha_v2（影子 Alpha v2）",
                "policy_v0_5": "policy_v0_5",
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
