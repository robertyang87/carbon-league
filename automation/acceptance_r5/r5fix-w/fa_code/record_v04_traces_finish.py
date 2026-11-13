"""record_v04_traces_finish: the main record_v04_traces run was
killed after 12/14 games (all 12 verified +0.00, files on disk).
This finisher records the 2 remaining ablV04notree games, verifies
them against eval_v0_4_results.jsonl, and appends the complete
v0.4 section (all 14 rows) to traces/INDEX.md."""

import json
import os
from concurrent.futures import ProcessPoolExecutor

import record_trace

HERE = os.path.dirname(os.path.abspath(__file__))

JOBS = [("policy_v0_4_notree.py", "reference_juzi", 1001, side,
         "ablV04notree") for side in (0, 1)]

ALL = ([("policy_v0_4", "eval_v04", "reference_juzi", s, side)
        for s in (1000, 1001, 1002, 1003) for side in (0, 1)]
       + [("policy_v0_4", "eval_v04_ex", "example", s, side)
          for s in (2000, 2001) for side in (0, 1)]
       + [("policy_v0_4_notree", "ablV04notree", "reference_juzi",
           1001, side) for side in (0, 1)])


def load_expected():
    exp = {}
    with open(os.path.join(HERE, "eval_v0_4_results.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                exp[(r["tag"], r["seed"], r["policy_side"])] = (
                    r["my_cash"], r["opp_cash"])
    return exp


def main():
    expected = load_expected()
    with ProcessPoolExecutor(max_workers=2) as ex:
        for r in ex.map(record_trace._job, JOBS):
            my = r["final_cash"][r["side"]]
            op = r["final_cash"][1 - r["side"]]
            emy, eop = expected[(r["tag"], r["seed"], r["side"])]
            ok = round(my - emy, 4) == 0.0 and round(op - eop, 4) == 0.0
            print(f"{r['tag']} seed{r['seed']} side{r['side']}: "
                  f"{my:.2f}:{op:.2f} {'OK' if ok else 'MISMATCH'}",
                  flush=True)
            assert ok
    lines = ["", "## v0.4 补录（policy_v0_4 / policy_v0_4_notree，2026-11-03）",
             "",
             "| 版本 | 对手 | seed | 先后手（我方） | 终分（我方 : 对手） "
             "| 原记录 | 核对差 | 文件 |",
             "|---|---|---|---|---|---|---|---|"]
    opp_name = {"reference_juzi": "juzi（reference_juzi）",
                "example": "example"}
    for pol, tag, opp, seed, side in ALL:
        my, op = expected[(tag, seed, side)]
        fname = f"{pol}_vs_{opp}_seed{seed}_side{side}.json.gz"
        assert os.path.exists(os.path.join(HERE, "traces", fname)), fname
        side_txt = "P0 先手" if side == 0 else "P1 后手"
        lines.append(
            f"| {pol}（{tag}） | {opp_name[opp]} | {seed} | {side_txt} "
            f"| {my:.2f} : {op:.2f} | {my:.2f} : {op:.2f} "
            f"| +0.00 / +0.00 | [{fname}]({fname}) |")
    lines.append("")
    with open(os.path.join(HERE, "traces", "INDEX.md"), "a") as f:
        f.write("\n".join(lines))
    print("INDEX section appended (14 rows).")


if __name__ == "__main__":
    main()
