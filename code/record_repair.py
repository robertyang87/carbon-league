"""record_repair: re-record every eval_integration game with the policy
file that ACTUALLY played it, verify 0.00 against
eval_integration_results.jsonl, and append the integration section to
traces/INDEX.md.

Why this exists: batches A_base / B_full were played by the original
full-stack fork_v0_2.py and F_base / F2_base by its (camp_fix=0)
incarnation, but the fork_v0_2.py filename was later overwritten by
the finalized all-off configuration, so a naive re-record under that
name replays the wrong policy (32 mismatches in the first attempt;
those traces were deleted). The historical files are preserved as
fork_v0_2_fullstack.py (byte-identical rebuild via
build_integration.apply_patches, 121,090 B) and
fork_v0_2_nocampfix.py, and are substituted here per batch.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import record_integration_traces as rit  # applies the loader/seed patches
import record_trace

HERE = os.path.dirname(os.path.abspath(__file__))

OVERRIDE = {
    "A_base": "fork_v0_2_fullstack.py",
    "B_full": "fork_v0_2_fullstack.py",
    "F_base": "fork_v0_2_nocampfix.py",
    "F2_base": "fork_v0_2_nocampfix.py",
}


def load_expected():
    exp, jobs = {}, []
    with open(os.path.join(HERE, "eval_integration_results.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["batch"], r["seed"], r["policy_side"])
            exp[key] = (r["my_cash"], r["opp_cash"])
            a_file = OVERRIDE.get(r["batch"], r["a_file"])
            jobs.append((a_file, os.path.splitext(r["b_file"])[0],
                         r["seed"], r["policy_side"], r["batch"]))
    return exp, jobs


def main():
    expected, jobs = load_expected()
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
        print("MISMATCHED:", [(r["tag"], r["seed"], r["side"]) for r in bad])

    lines = ["", "## 集成线 fork-v0.2（eval_integration，2026-11-03）", "",
             "注：A_base / B_full 的 A 方为原始全栈版 fork_v0_2.py，"
             "其字节级重建保存为 fork_v0_2_fullstack.py（121,090 B）；"
             "F_base / F2_base 的 A 方为其 campfix 关断版，保存为 "
             "fork_v0_2_nocampfix.py。定版 fork_v0_2.py 的文件名曾被"
             "覆盖，直接以该名重录会放错策略（首轮 32 局 MISMATCH，"
             "坏 trace 已删除后按本表文件重录）。",
             "",
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
