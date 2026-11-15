#!/bin/bash
# Runs directly on h24 via cloud assistant (READ ONLY): m3s2r1_e0 live progress.
MARK="M3H24E0P5T7"
echo "MARKER $MARK START"
F=/root/carbon/data/bigeval_live/m3s2r1_e0.jsonl
ls -la $F
echo "--- WC ---"
timeout 30 wc -l $F
echo "--- HEAD1 ---"
timeout 30 head -1 $F
echo "--- TAIL3 ---"
timeout 30 tail -3 $F
echo "--- PYAGG ---"
timeout 60 python3 - "$F" <<'PYEOF'
import json, sys
from collections import Counter
path = sys.argv[1]
rows = []
bad = 0
for line in open(path):
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except Exception:
        bad += 1
print("parsed", len(rows), "bad", bad)
if rows:
    print("keys", sorted(rows[0].keys()))
    for k in ("margin", "diff", "score_diff", "delta", "net"):
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if vals:
            wins = sum(1 for v in vals if v > 0)
            print("FIELD", k, "n", len(vals), "wins_pos", wins, "mean", round(sum(vals) / len(vals), 2))
    for k in ("win", "won", "a_win"):
        vals = [r[k] for r in rows if k in r]
        if vals:
            print("FIELD", k, "true_count", sum(1 for v in vals if v))
    vals = [r["winner"] for r in rows if "winner" in r]
    if vals:
        print("FIELD winner counts", dict(Counter(str(v) for v in vals)))
PYEOF
echo "MARKER $MARK END"
