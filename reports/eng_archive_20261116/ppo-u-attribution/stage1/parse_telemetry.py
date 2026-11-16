#!/usr/bin/env python3
"""Stage 1 deliverable B: parse train.jsonl telemetry for it74-it95.

Reads the local copy of the run's train.jsonl (copied from L20
/root/carbon/data/rl3/snipe_n1/train.jsonl, md5 recorded in RECEIPT),
emits a CSV with every scalar field actually present, marking the
it82 boundary (r19 / S19-A utility-source=checkpoint switch) and the
it78 boundary (vhead swap, machine fact).
"""
import csv
import json
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "train.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "telemetry_it74_95.csv"

rows = []
all_keys = []
with open(SRC) as f:
    for line in f:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "it" not in r or "games" not in r:
            continue
        rows.append(r)
        for k, v in r.items():
            if isinstance(v, (int, float, bool)) and k not in all_keys:
                all_keys.append(k)

its = sorted(r["it"] for r in rows)
print(f"rows={len(rows)} it_range={its[0]}..{its[-1]}")
print("scalar fields:", ",".join(all_keys))

sel = [r for r in rows if 74 <= r["it"] <= 95]
sel.sort(key=lambda r: r["it"])
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["it"] + [k for k in all_keys if k != "it"] + ["note"])
    for r in sel:
        note = ""
        if r["it"] == 78:
            note = "vhead swap fired at boundary_it=78 (machine fact)"
        if r["it"] == 82:
            note = "S19-A r19 switch at boundary_it=82 (PREREG marker)"
        w.writerow([r["it"]] + [r.get(k, "") for k in all_keys if k != "it"] + [note])
print(f"wrote {OUT} rows={len(sel)}")
