"""Full validation of Studio S3 shard (all 8 lanes) against trial profile."""
import json
import os
from collections import Counter

BASE = "/Users/arm/junyang/carbon/data/s3_enum_studio/out"
rows = []
per_lane = {}
for lane in range(8):
    name = f"lane{lane}"
    p = os.path.join(BASE, name, f"roots_{name}.jsonl")
    n = 0
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
            n += 1
    per_lane[name] = n
    rep = json.load(open(os.path.join(BASE, name, f"report_{name}.json"),
                         encoding="utf-8"))
    print(name, "report:", {k: rep[k] for k in
          ("games_played", "producing_games_N", "zero_root_games_M",
           "total_roots", "settlement_decidable_roots",
           "settlement_argmax_nonzero", "trigger_events_1_4")},
          "jsonl_rows:", n)

print(f"TOTAL rows: {len(rows)}")
ids = [r["root_id"] for r in rows]
print(f"root_id unique: {len(set(ids))}/{len(ids)}")
games = sorted({r["game"] for r in rows})
print("games union:", games, "count:", len(games),
      "exact 270-299:", games == list(range(270, 300)))
print("seeds:", sorted({r["seed"] for r in rows}) == list(range(986470, 986500)))
print("seats:", Counter(r["seat"] for r in rows))
print("steps: min", min(r["step"] for r in rows), "max", max(r["step"] for r in rows))
print("trigger_class:", Counter(r.get("trigger_class") for r in rows))
print("sub_direction:", Counter(r.get("sub_direction") for r in rows))
print("cand_count:", Counter(r.get("cand_count") for r in rows))
dec = [r for r in rows if r.get("settlement_decidable")]
print(f"decidable: {len(dec)}; steps:", sorted({r['step'] for r in dec}),
      "argmax:", Counter(r.get("settlement_argmax") for r in dec))
print("per-game:", sorted(Counter(r["game"] for r in rows).items()))
# witness sizes + register totals
for lane in range(8):
    name = f"lane{lane}"
    w = os.path.join(BASE, name, f"witness_{name}.jsonl")
    rj = os.path.join(BASE, name, f"roots_{name}.json")
    print(name, "witness_bytes:", os.path.getsize(w),
          "roots_json_bytes:", os.path.getsize(rj))
