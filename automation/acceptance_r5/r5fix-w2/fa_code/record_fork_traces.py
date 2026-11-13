"""record_fork_traces: record all eval_fork games with record_trace.py,
verify each re-run's final cash against eval_fork_results.jsonl
(exact 0.00 required), and append the fork-tuning section to
traces/INDEX.md.

Same reproducibility protocol as record_v53_traces.py: eval_fork reuses
eval_v53.play_game, which seeds random.seed(7919*seed + 104729*side + 31)
as its first statement; record_game is wrapped with the identical seed,
and record_trace.load_agent_fn is monkeypatched to eval_v53's loader
rule (policy_* -> make_agent(), else module agent()).
"""

import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import eval_v53
import record_trace

HERE = os.path.dirname(os.path.abspath(__file__))

record_trace.load_agent_fn = lambda path, kind: eval_v53.load_agent_fn(path)

_orig_record_game = record_trace.record_game


def _seeded_record_game(policy_file, opp_kind, seed, policy_side, tag=None):
    random.seed(eval_v53.rng_seed(seed, policy_side))
    return _orig_record_game(policy_file, opp_kind, seed, policy_side, tag)


record_trace.record_game = _seeded_record_game

TAGS = ("rho004", "rho006", "rho008", "rho010",
        "v53shadow", "forkshadow", "forkbase")


def load_jobs():
    jobs, exp = [], {}
    with open(os.path.join(HERE, "eval_fork_results.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["tag"] not in TAGS:
                continue
            key = (r["tag"], r["seed"], r["policy_side"])
            if key in exp:
                continue
            exp[key] = (r["my_cash"], r["opp_cash"])
            jobs.append((r["a_file"],
                         os.path.splitext(r["b_file"])[0],
                         r["seed"], r["policy_side"], r["tag"]))
    return jobs, exp


def main():
    jobs, expected = load_jobs()
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

    names = {
        "fork_rho004": "fork rho=0.04", "fork_rho006": "fork rho=0.06",
        "fork_rho008": "fork rho=0.08", "fork_rho010": "fork rho=0.10",
        "fork_v0_1": "fork-v0.1（rho=0.08）", "base_v53": "base_v53（基座）",
        "shadow_alpha_v2": "shadow_alpha_v2（影子 Alpha v2）",
        "reference_juzi": "juzi（reference_juzi）",
    }
    lines = ["", "## v53 基座调优轮（eval_fork，2026-11-03）", "",
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
    print("INDEX.md fork section appended.")


if __name__ == "__main__":
    main()
