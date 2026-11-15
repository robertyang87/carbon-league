#!/usr/bin/env python3
"""War-room-sync 13:35 trip 2026-11-15 (v142): batched GitHub API branch freshness check (read-only).
Calls: main x1, sync/h29 + sync/h34 x2, rate_limit x1. h23/h24 already re-read 13:12/13:14 by probes (unchanged); h28/h31 archived, unchanged for days, not re-swept per API discipline."""
import sys, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {}
try:
    cs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha=main&per_page=1")
    c = cs[0]
    out["main"] = {"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"],
                   "msg": c["commit"]["message"].split("\n")[0][:90]}
except Exception as e:
    out["main_error"] = repr(e)
for b in ["sync/h29", "sync/h34"]:
    try:
        cs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={b}&per_page=1")
        c = cs[0]
        out[b] = {"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"]}
    except Exception as e:
        out[b] = {"error": repr(e)}
try:
    rl = gh_api.req("GET", "/rate_limit")
    out["rate_core_remaining"] = rl["resources"]["core"]["remaining"]
    out["rate_core_reset"] = rl["resources"]["core"]["reset"]
except Exception as e:
    out["rate_error"] = repr(e)
print(json.dumps(out, indent=1, ensure_ascii=False))
