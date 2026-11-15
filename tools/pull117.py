#!/usr/bin/env python3
"""Pull full match list for a strategy from the 117.72.210.134 endpoint (full view).
Usage: python3 pull117.py <strategy_id> <out.json>
Paginates limit=100 until next_offset is empty. Uses curl (reliable vs IncompleteRead).
"""
import json, subprocess, sys

BASE = "https://117.72.210.134/api/matches?compact=1&limit=100&offset={}&strategy={}"

def fetch(url):
    r = subprocess.run(["curl", "-skL", "--max-time", "60", url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl rc={r.returncode}: {r.stderr[:300]}")
    return json.loads(r.stdout)

def main():
    sid, out = sys.argv[1], sys.argv[2]
    all_m, off, pages = [], 0, 0
    while True:
        d = fetch(BASE.format(off, sid))
        ms = d.get("matches", [])
        all_m.extend(ms)
        pages += 1
        nxt = d.get("next_offset")
        if not ms or nxt in (None, "", 0):
            break
        off = int(nxt)
        if pages > 30:
            raise RuntimeError("too many pages")
    json.dump(all_m, open(out, "w"), ensure_ascii=False)
    print(f"{sid}: {len(all_m)} matches, {pages} pages -> {out}")

if __name__ == "__main__":
    main()
