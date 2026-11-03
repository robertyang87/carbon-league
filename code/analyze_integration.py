"""analyze_integration: aggregate eval_integration_results.jsonl.

Usage: python3 analyze_integration.py <batch> [<batch> ...]
Prints per batch: n, mean margin, W/L, per-seed pair sums, worst game,
mean ledger deltas (A minus B) for the scorecard items, mean ms/step.
"""

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LED_KEYS = ("deposits", "tree_income", "plant_count", "plant_spend",
            "seize_count", "seize_spend", "deaths", "death_cargo",
            "stranded", "recruit_count")


def load():
    rows = []
    with open(os.path.join(HERE, "eval_integration_results.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(batch, rows):
    rs = [r for r in rows if r["batch"] == batch]
    if not rs:
        print(f"[{batch}] no rows")
        return
    rs.sort(key=lambda r: (r["seed"], r["policy_side"]))
    margins = [r["margin"] for r in rs]
    seeds = {}
    for r in rs:
        seeds.setdefault(r["seed"], []).append(r["margin"])
    print(f"== {batch}: {rs[0]['a_file']} (A) vs {rs[0]['b_file']} (B), "
          f"n={len(rs)}")
    for r in rs:
        print(f"   seed{r['seed']} side{r['policy_side']}: "
              f"{r['my_cash']:.2f} : {r['opp_cash']:.2f}  "
              f"margin {r['margin']:+.2f}")
    print(f"   mean {statistics.mean(margins):+.2f}  "
          f"median {statistics.median(margins):+.2f}  "
          f"stdev {statistics.pstdev(margins):.1f}  "
          f"W{sum(1 for m in margins if m > 0)}/L{sum(1 for m in margins if m < 0)}")
    print(f"   per-seed pair sums: " +
          ", ".join(f"{s}:{sum(v):+.1f}" for s, v in sorted(seeds.items())))
    worst = min(rs, key=lambda r: r["margin"])
    print(f"   worst game: seed{worst['seed']} side{worst['policy_side']} "
          f"{worst['margin']:+.2f}")
    print(f"   a_ms/step mean {statistics.mean(r['a_ms_per_step'] for r in rs):.1f}, "
          f"worst step max {max(r['a_worst_step_ms'] for r in rs):.0f}")
    print("   ledger deltas (A-B, per game mean):")
    for k in LED_KEYS:
        d = statistics.mean(r["ledger_pol"][k] - r["ledger_opp"][k]
                            for r in rs)
        print(f"     {k:14s} {d:+.1f}")


if __name__ == "__main__":
    rows = load()
    for b in sys.argv[1:]:
        summarize(b, rows)
