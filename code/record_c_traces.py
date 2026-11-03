"""record_c_traces: record every eval_c_pricing game with
record_trace.py, verify each re-run's final cash against
eval_c_pricing_results.jsonl (exact 0.00 required), and append the
case-C section to traces/INDEX.md. Same reproducibility protocol as
record_probe_traces.py (seeded wrapper + eval_v53 loader rule).
"""

import json
import os
import random
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


REMAP = {  # a_file as logged at eval time -> file holding that exact config now
    ("call", "fork_c_pricing.py"): "fork_c_all.py",
    ("callv05", "fork_c_pricing.py"): "fork_c_all.py",
    ("cmirror", "fork_c_pricing.py"): "fork_c_all.py",
    ("vshadow", "fork_c_pricing.py"): "fork_c_nosoft.py",
    ("vmirror", "fork_c_pricing.py"): "fork_c_nosoft.py",
}


def load_jobs():
    jobs = []
    exp = {}
    for fn in ("eval_c_pricing_results.jsonl",
               "eval_c_verify_results.jsonl",
               "eval_c_final_results.jsonl"):
        with open(os.path.join(HERE, fn)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                opp = r["b_file"][:-3] if r["b_file"].endswith(".py") \
                    else r["b_file"]
                a_file = REMAP.get((r["tag"], r["a_file"]), r["a_file"])
                jobs.append((a_file, opp, r["seed"], r["policy_side"],
                             r["tag"]))
                exp[(r["tag"], r["seed"], r["policy_side"])] = (
                    r["my_cash"], r["opp_cash"])
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
        return

    names = {
        "fork_c_pricing": "fork_c_pricing（案C 交付版·全关回滚后）",
        "fork_c_all": "fork_c_all（案C 定价化全开·存档）",
        "fork_c_nosoft": "fork_c_nosoft（案C soft 回滚后·lethal+minreach 开）",
        "fork_c_soft": "fork_c_soft（案C 消融 soft 单开）",
        "fork_c_lethal": "fork_c_lethal（案C 消融 lethal 单开）",
        "fork_c_off": "fork_c_off（案C 全关 parity）",
        "base_v53": "base_v53（公开源码基座）",
        "shadow_alpha_v2": "影子 Alpha v2（shadow_alpha_v2）",
        "policy_v0_5": "我方 v0.5（policy_v0_5）",
    }
    lines = ["", "## 结构案 C 定价化验收（eval_c_pricing，2026-11-03）", "",
             "| 版本 | 对手 | seed | 先后手（我方） | 终分（我方 : 对手） "
             "| 原记录 | 核对差 | 文件 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        side_txt = "P0 先手" if r["side"] == 0 else "P1 后手"
        pol = r["policy"][:-3] if r["policy"].endswith(".py") \
            else r["policy"]
        lines.append(
            f"| {names.get(pol, pol)}（{r['tag']}） "
            f"| {names.get(r['opp'], r['opp'])} "
            f"| {r['seed']} | {side_txt} "
            f"| {r['my_cash']:.2f} : {r['opp_cash']:.2f} "
            f"| {r['exp_my']:.2f} : {r['exp_opp']:.2f} "
            f"| {r['diff_my']:+.2f} / {r['diff_opp']:+.2f} "
            f"| [{r['file']}]({r['file']}) |")
    lines.append("")
    with open(os.path.join(HERE, "traces", "INDEX.md"), "a") as f:
        f.write("\n".join(lines))
    print("INDEX.md case-C section appended.")


if __name__ == "__main__":
    main()
