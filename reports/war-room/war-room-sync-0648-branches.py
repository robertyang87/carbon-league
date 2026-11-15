#!/usr/bin/env python3
"""War-room-sync 06:48 trip: batched GitHub API branch freshness check (read-only).
Calls: main commits x1, 3 sync-branch heads, sync/h29 summaries listing, rate_limit x1."""
import sys, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {}
try:
    cs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha=main&per_page=10")
    out["main"] = [{"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"],
                    "msg": c["commit"]["message"].split("\n")[0][:90]} for c in cs]
except Exception as e:
    out["main_error"] = repr(e)
for b in ["sync/h24", "sync/h28", "sync/h31"]:
    try:
        cs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={b}&per_page=1")
        c = cs[0]
        out[b] = {"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"]}
    except Exception as e:
        out[b] = {"error": repr(e)}
try:
    items = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/synced/h29/summaries?ref=sync/h29")
    out["h29_summaries"] = sorted([i["name"] for i in items if i["type"] == "file"])
except Exception as e:
    out["h29_summaries_error"] = repr(e)
try:
    rl = gh_api.req("GET", "/rate_limit")
    out["rate_core_remaining"] = rl["resources"]["core"]["remaining"]
    out["rate_core_reset"] = rl["resources"]["core"]["reset"]
except Exception as e:
    out["rate_error"] = repr(e)
print(json.dumps(out, indent=1, ensure_ascii=False))
