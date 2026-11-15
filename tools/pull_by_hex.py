#!/usr/bin/env python3
"""Pull a strategy's full match list from the 117 endpoint by hex strategy id.

The ladder site changed its /api/matches filter: `strategy=<display name>`
now 400s ("Invalid pagination parameters"); the filter is now
`strategy=<hex strategy id>` (the id shown in `ladder.py board` output).
Usage: pull_by_hex.py <hex_id> <out.json>
"""
import json, ssl, sys, time, urllib.request, urllib.parse

BASE = "https://117.72.210.134"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def fetch(off, sid):
    q = urllib.parse.urlencode({"compact": 1, "limit": 100, "offset": off, "strategy": sid})
    req = urllib.request.Request(BASE + "/api/matches?" + q, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
        return json.loads(r.read())

def main():
    sid, out = sys.argv[1], sys.argv[2]
    got, off = [], 0
    while True:
        d = fetch(off, sid)
        got += d.get("matches", [])
        nxt = d.get("next_offset")
        if nxt is None:
            break
        off = nxt
        time.sleep(0.5)
    seen, uniq = set(), []
    for m in got:
        k = (m.get("id"), m.get("game"))
        if k not in seen:
            seen.add(k)
            uniq.append(m)
    json.dump(uniq, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"{len(uniq)} games -> {out}")

if __name__ == "__main__":
    main()
