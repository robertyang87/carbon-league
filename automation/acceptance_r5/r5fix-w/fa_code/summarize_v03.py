"""summarize_v03: aggregate eval_v0_1_results.jsonl rows for the v0.3
report -- per-game table for eval_v03, cross-version comparison
(v1 diag_v1 / v0.1 eval_v01 / v0.2 eval_v02 / v0.3 eval_v03), ledger
means, the seed-1001-P0 ablation pair, and example sanity rows."""

import json
import os
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = []
with open(os.path.join(HERE, "eval_v0_1_results.jsonl")) as f:
    for line in f:
        line = line.strip()
        if line:
            ROWS.append(json.loads(line))


def get(tag):
    return sorted([r for r in ROWS if r["tag"] == tag],
                  key=lambda r: (r["seed"], r["policy_side"]))


def ledger_mean(rows, who):
    keys = ["deposits", "tree_income", "recruit_spend", "recruit_count",
            "plant_spend", "plant_count", "seize_spend", "seize_count",
            "deaths", "death_cargo", "stranded"]
    return {k: round(mean(r[f"ledger_{who}"][k] for r in rows), 1)
            for k in keys}


for tag in ("eval_v03",):
    rows = get(tag)
    print(f"== {tag}: {len(rows)} games ==")
    for r in rows:
        L, O = r["ledger_pol"], r["ledger_opp"]
        print(f"seed{r['seed']} P{r['policy_side']}  {r['my_cash']:.0f}:"
              f"{r['opp_cash']:.0f}  margin {r['margin']:+.0f}  "
              f"seize {L['seize_count']:.0f}/{O['seize_count']:.0f}  "
              f"stranded {L['stranded']:.0f}  ms/step {r['my_ms_per_step']}")
    if rows:
        print(f"mean margin {mean(r['margin'] for r in rows):+.1f}  "
              f"my {mean(r['my_cash'] for r in rows):.0f} vs "
              f"opp {mean(r['opp_cash'] for r in rows):.0f}  "
              f"wins {sum(r['win'] for r in rows)}/{len(rows)}  "
              f"ms/step {mean(r['my_ms_per_step'] for r in rows):.1f}  "
              f"wall max {max(r['wall_s'] for r in rows):.0f}s")
        print("ledger_pol mean: ", ledger_mean(rows, "pol"))
        print("ledger_opp mean: ", ledger_mean(rows, "opp"))

print("\n== cross-version (vs juzi, 8 paired games each) ==")
for tag in ("diag_v1", "eval_v01", "eval_v02", "eval_v03"):
    rows = get(tag)
    if len(rows) != 8:
        print(f"{tag}: only {len(rows)} rows")
        continue
    mp, mo = ledger_mean(rows, "pol"), ledger_mean(rows, "opp")
    print(f"{tag}: margin {mean(r['margin'] for r in rows):+7.0f} | "
          f"dep {mp['deposits']:.0f}/{mo['deposits']:.0f} "
          f"({mp['deposits'] - mo['deposits']:+.0f}) | "
          f"tree {mp['tree_income']:.0f}/{mo['tree_income']:.0f} "
          f"({mp['tree_income'] - mo['tree_income']:+.0f}) | "
          f"seize {mp['seize_count']:.1f}/{mo['seize_count']:.1f} | "
          f"plant {mp['plant_count']:.1f}/{mo['plant_count']:.1f} | "
          f"deaths {mp['deaths']:.1f}/{mo['deaths']:.1f} "
          f"deathcargo {mp['death_cargo']:.0f}/{mo['death_cargo']:.0f} | "
          f"recruit {mp['recruit_count']:.1f}/{mo['recruit_count']:.1f} | "
          f"stranded {mp['stranded']:.0f} | "
          f"ms/step {mean(r['my_ms_per_step'] for r in rows):.1f}")

print("\n== ablation seed1001 P0 ==")
for tag in ("eval_v03", "ablV03notree"):
    for r in get(tag):
        if r["seed"] == 1001 and r["policy_side"] == 0:
            L = r["ledger_pol"]
            print(f"{tag}: {r['my_cash']:.0f}:{r['opp_cash']:.0f} "
                  f"margin {r['margin']:+.0f} dep {L['deposits']:.0f} "
                  f"tree {L['tree_income']:.0f} seize {L['seize_count']:.0f} "
                  f"plant {L['plant_count']:.0f} deaths {L['deaths']:.0f} "
                  f"deathcargo {L['death_cargo']:.0f}")

print("\n== example sanity ==")
for tag in ("eval_v03_ex", "smoke_v03"):
    rows = get(tag)
    for r in rows:
        print(f"{tag} seed{r['seed']} P{r['policy_side']}: "
              f"{r['my_cash']:.0f}:{r['opp_cash']:.0f} "
              f"margin {r['margin']:+.0f} stranded "
              f"{r['ledger_pol']['stranded']:.0f} "
              f"ms/step {r['my_ms_per_step']}")
