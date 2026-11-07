"""s2_summary -- summarize evo_results.jsonl rows by tag: n, wins, mean +- se (game-level and seed-paired),
worst game, worst step ms, death cargo.  Against plain v9 on the same seeds both sides, v9-vs-v9 is
exactly 0 per seed pair, so the mean margin is the improvement over v9.

python3 s2_summary.py TAG [TAG ...]
"""
import json
import os
import statistics as st
import sys
from collections import defaultdict

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "evo_results.jsonl")


def main(tags):
    rows = [json.loads(l) for l in open(RES)]
    for tag in tags:
        R = [r for r in rows if r.get("tag") == tag and "margin" in r]
        if not R:
            print(tag, "no rows"); continue
        m = [r["margin"] for r in R]
        by = defaultdict(list)
        for r in R:
            by[r["seed"]].append(r["margin"])
        pm = [sum(v) / 2 for v in by.values() if len(v) == 2]
        out = {"tag": tag, "a": R[0]["a_file"], "n": len(m), "wins": sum(1 for r in R if r["win"]),
               "mean": round(st.mean(m), 1), "se": round(st.stdev(m) / len(m) ** 0.5, 1) if len(m) > 1 else None,
               "pairs": len(pm), "pair_se": round(st.stdev(pm) / len(pm) ** 0.5, 1) if len(pm) > 1 else None,
               "worst": min(m), "median": st.median(m),
               "worst_step_ms": max(r.get("a_worst_step_ms", 0) for r in R),
               "a_ms": round(st.mean(r["a_ms_per_step"] for r in R), 1),
               "death_cargo_pol": round(st.mean(r.get("pol_death_cargo") or 0 for r in R)),
               "death_cargo_opp": round(st.mean(r.get("opp_death_cargo") or 0 for r in R))}
        print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1:])
