"""Validate Studio S3 pilot outputs against the trial profile (read-only)."""
import json
import os
import sys
from collections import Counter

BASE = "/Users/arm/junyang/carbon/data/s3_enum_studio/out"
rows = []
for lane in ("lane0", "lane1"):
    p = os.path.join(BASE, lane, f"roots_{lane}.jsonl")
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))

print(f"rows: {len(rows)}")
ids = [r["root_id"] for r in rows]
print(f"root_id unique: {len(set(ids))}/{len(ids)}")
print("games:", sorted({r["game"] for r in rows}))
print("seeds:", sorted({r["seed"] for r in rows}))
print("seats:", Counter(r["seat"] for r in rows))
print("steps: min", min(r["step"] for r in rows), "max", max(r["step"] for r in rows))
print("trigger_class:", Counter(r.get("trigger_class") for r in rows))
print("sub_direction:", Counter(r.get("sub_direction") for r in rows))
print("cand_count:", Counter(r.get("cand_count") for r in rows))
dec = [r for r in rows if r.get("settlement_decidable")]
print(f"decidable: {len(dec)}; steps:", sorted({r['step'] for r in dec}),
      "argmax:", Counter(r.get("settlement_argmax") for r in dec))
# per-game counts
print("per-game:", sorted(Counter(r["game"] for r in rows).items()))
# timing breakdown from reports
for lane in ("lane0", "lane1"):
    rep = json.load(open(os.path.join(BASE, lane, f"report_{lane}.json"), encoding="utf-8"))
    print(lane, "report:", {k: rep[k] for k in
          ("games_played", "producing_games_N", "zero_root_games_M",
           "total_roots", "settlement_decidable_roots",
           "settlement_argmax_nonzero", "trigger_events_1_4")})
    t = rep.get("timing", {})
    if t:
        ac, rp = t.get("agent_call_s", []), t.get("root_proc_s", [])
        if ac:
            print(f"  agent_call_s: n={len(ac)} mean={sum(ac)/len(ac):.4f} max={max(ac):.4f}")
        if rp:
            print(f"  root_proc_s: n={len(rp)} mean={sum(rp)/len(rp):.4f} max={max(rp):.4f}")
