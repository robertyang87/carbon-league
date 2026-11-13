"""record_raid: backfill traces for all raid-line eval games.

Re-runs every game recorded in eval_raid_results.jsonl (tags eval_raid,
abl_raid_nohunt, eval_raid_ex) through record_trace.py's recorder,
verifies the re-run final cash against the eval record (deterministic
reproduction: the diff must be 0.00), and appends a raid section to
traces/INDEX.md.  Smoke-tagged games are skipped.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import record_trace as rt  # noqa: E402

RESULTS = os.path.join(HERE, "eval_raid_results.jsonl")
TAG_POLICY = {
    "eval_raid": "policy_raid_v0_1.py",
    "eval_raid_ex": "policy_raid_v0_1.py",
    "abl_raid_nohunt": "policy_raid_v0_1_nohunt.py",
}
OPP_NAME = {"reference_juzi": "juzi（reference_juzi）", "example": "example"}


def main():
    expected = {}
    with open(RESULTS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["tag"] not in TAG_POLICY:
                continue
            expected[(r["tag"], r["seed"], r["policy_side"])] = r
    jobs = []
    for (tag, seed, side), r in sorted(expected.items()):
        jobs.append((TAG_POLICY[tag], r["opp"], seed, side, tag))
    print(f"{len(jobs)} raid games to record", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(rt._job, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            exp = expected[(r["tag"], r["seed"], r["side"])]
            my = r["final_cash"][r["side"]]
            op = r["final_cash"][1 - r["side"]]
            r["my_cash"], r["opp_cash"] = my, op
            r["exp_my"], r["exp_opp"] = exp["my_cash"], exp["opp_cash"]
            r["diff_my"] = round(my - exp["my_cash"], 4)
            r["diff_opp"] = round(op - exp["opp_cash"], 4)
            r["ok"] = abs(r["diff_my"]) <= 0.02 and abs(r["diff_opp"]) <= 0.02
            results.append(r)
            print(f"{r['tag']} seed{r['seed']} side{r['side']}: "
                  f"rerun {my:.2f}:{op:.2f} vs recorded "
                  f"{exp['my_cash']:.2f}:{exp['opp_cash']:.2f} "
                  f"diff {r['diff_my']:+.2f}/{r['diff_opp']:+.2f} "
                  f"{'OK' if r['ok'] else 'MISMATCH'}", flush=True)
    results.sort(key=lambda r: (r["tag"], r["seed"], r["side"]))
    bad = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(bad)}/{len(results)} games match.")
    if bad:
        print("MISMATCHED:", [(r["tag"], r["seed"], r["side"]) for r in bad])
        return 1
    lines = ["", "## 突袭线 raid v0.1 补录（policy_raid_v0_1，2026-11-03）", "",
             "| 版本 | 对手 | seed | 先后手（我方） | 终分（我方 : 对手） "
             "| 原记录 | 核对差 | 文件 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        side_txt = "P0 先手" if r["side"] == 0 else "P1 后手"
        pol_name = r["policy"]
        lines.append(
            f"| {pol_name}（{r['tag']}） | {OPP_NAME[r['opp']]} "
            f"| {r['seed']} | {side_txt} "
            f"| {r['my_cash']:.2f} : {r['opp_cash']:.2f} "
            f"| {r['exp_my']:.2f} : {r['exp_opp']:.2f} "
            f"| {r['diff_my']:+.2f} / {r['diff_opp']:+.2f} "
            f"| [{r['file']}]({r['file']}) |")
    lines.append("")
    with open(os.path.join(rt.TRACES_DIR, "INDEX.md"), "a") as f:
        f.write("\n".join(lines))
    print("INDEX.md raid section appended.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
