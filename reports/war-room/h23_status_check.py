#!/usr/bin/env python3
"""h23 status check: read sync/h23 branch head, synced/h23 listing,
bigeval_live listing (newest live files + line counts via blob size proxy is
NOT enough -- line counts need file content; fetch only the newest few),
and *_cal.out full contents. Also reads sync/h29 head for the neighbor C1 line.
Usage: python3 h23_status_check.py [--cal]  (cal fetches cal.out contents)
Prints JSON to stdout. Uses gh_api.req (shared GitHub API pool: keep calls few).
"""
import json, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"

def branch_head(branch):
    b = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{branch}")
    c = b["commit"]["commit"]
    return {"branch": branch, "sha": b["commit"]["sha"][:8],
            "message": c["message"].splitlines()[0],
            "date": c["committer"]["date"]}

def listing(path, ref):
    items = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={ref}")
    if isinstance(items, dict):
        return {"error": items.get("message"), "path": path}
    return [{"name": i["name"], "type": i["type"], "size": i.get("size", 0),
             "sha": i["sha"][:8]} for i in items]

def file_text(path, ref):
    f = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={ref}")
    if isinstance(f, dict) and "content" not in f:
        return {"error": f.get("message")}
    raw = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
    return {"size": f["size"], "sha": f["sha"][:8], "text": raw}

out = {}
out["rate_limit"] = gh_api.req("GET", "/rate_limit")["resources"]["core"]
out["h23_head"] = branch_head("sync/h23")
out["h29_head"] = branch_head("sync/h29")
h23 = listing("synced/h23", "sync/h23")
out["synced_h23_count"] = len(h23) if isinstance(h23, list) else h23
if isinstance(h23, list):
    cals = [i for i in h23 if i["name"].endswith("_cal.out")]
    out["cal_files"] = [i["name"] for i in cals]
    if "--cal" in sys.argv:
        out["cal_contents"] = {i["name"]: file_text(f"synced/h23/{i['name']}", "sync/h23")
                               for i in cals}
live = listing("synced/h23/bigeval_live", "sync/h23")
out["bigeval_live_count"] = len(live) if isinstance(live, list) else live
if isinstance(live, list) and live:
    # newest 5 by name (live files are timestamp/seed named); fetch smallest metadata only
    out["bigeval_live_newest"] = sorted(live, key=lambda x: x["name"])[-5:]
print(json.dumps(out, indent=1, ensure_ascii=False))
