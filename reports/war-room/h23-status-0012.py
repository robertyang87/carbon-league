#!/usr/bin/env python3
"""H23 status check driver — run 00:12, 2026-11-14. Reads sync/h23, sync/h29, main heads,
synced/h23 + bigeval_live item counts, and the *_cal.out files, via gh_api.req."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {}

def branch_head(branch):
    b = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{branch}")
    c = b["commit"]["commit"]
    return {"sha": b["commit"]["sha"][:8], "date_utc": c["committer"]["date"],
            "message": c["message"].splitlines()[0][:100]}

for br in ("sync/h23", "sync/h29", "main"):
    try:
        out[br] = branch_head(br)
    except Exception as e:
        out[br] = {"error": str(e)}

def list_dir(path, ref):
    try:
        items = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={ref}")
        if isinstance(items, dict):
            return {"error": items.get("message", str(items))[:120]}
        return {"count": len(items),
                "names": [i["name"] for i in items]}
    except Exception as e:
        return {"error": str(e)[:200]}

h23 = list_dir("synced/h23", "sync/h23")
out["synced_h23"] = {"count": h23.get("count"), "names": h23.get("names", [])[:50],
                     "error": h23.get("error")}
live = list_dir("synced/h23/bigeval_live", "sync/h23")
out["bigeval_live"] = {"count": live.get("count"), "error": live.get("error")}
# newest live file names (by name sort, tail) for a rough recency hint
if live.get("names"):
    out["bigeval_live"]["tail_names"] = sorted(live["names"])[-5:]

cal = {}
for name in (h23.get("names") or []):
    if name.endswith("_cal.out"):
        try:
            f = gh_api.get_file(OWNER, REPO, f"synced/h23/{name}", ref="sync/h23")
            raw = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
            cal[name] = {"size": f["size"], "content": raw}
        except Exception as e:
            cal[name] = {"error": str(e)[:200]}
out["cal_out"] = cal

try:
    rl = gh_api.req("GET", "/rate_limit")
    core = rl["resources"]["core"]
    out["api_core"] = {"remaining": core["remaining"], "reset_utc": core["reset"]}
except Exception as e:
    out["api_core"] = {"error": str(e)[:120]}

print(json.dumps(out, indent=1, ensure_ascii=False))
