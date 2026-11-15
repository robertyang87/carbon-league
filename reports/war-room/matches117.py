#!/usr/bin/env python3
"""Pull full match list for a strategy from the 117 endpoint (full view),
paginating until next_offset is empty. Writes JSON to stdout path arg.
Usage: matches117.py <strategy_id> <out.json>
"""
import json, subprocess, sys

sid, out = sys.argv[1], sys.argv[2]
all_matches = []
offset = 0
while True:
    url = (f"https://117.72.210.134/api/matches?compact=1&limit=100"
           f"&offset={offset}&strategy={sid}")
    r = subprocess.run(["curl", "-skL", "--max-time", "60", url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"curl failed rc={r.returncode} offset={offset}: {r.stderr[:200]}",
              file=sys.stderr)
        sys.exit(2)
    try:
        data = json.loads(r.stdout)
    except Exception as e:
        print(f"bad json at offset={offset}: {e}; body head: {r.stdout[:200]}",
              file=sys.stderr)
        sys.exit(3)
    items = data.get("matches") or data.get("items") or []
    if isinstance(data, list):
        items = data
    all_matches.extend(items)
    nxt = data.get("next_offset") if isinstance(data, dict) else None
    if nxt is None or nxt == "" or nxt == offset:
        break
    offset = nxt
json.dump(all_matches, open(out, "w"), ensure_ascii=False)
print(f"total={len(all_matches)}")
