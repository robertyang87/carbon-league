#!/usr/bin/env python3
"""H24/C2 status check driver: read sync/h24, sync/h31, sync/h29, main branch heads
and directory listings via gh_api (shared GitHub API quota pool - keep calls minimal)."""
import sys, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"

def branch_head(branch):
    r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{branch}")
    c = r["commit"]["commit"]
    return {"sha": r["commit"]["sha"][:8], "date": c["committer"]["date"],
            "msg": c["message"].splitlines()[0][:100]}

def list_dir(path, ref):
    r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={ref}")
    if isinstance(r, dict):  # error or single file
        return r
    return [{"name": e["name"], "size": e.get("size", 0), "type": e["type"]} for e in r]

out = {}
for b in ["sync/h24", "sync/h31", "sync/h29", "main"]:
    try:
        out[b] = branch_head(b)
    except Exception as e:
        out[b] = {"error": str(e)[:200]}

# h24 dir listings (counts only, cheap)
for path, ref in [("synced/h24", "sync/h24"), ("synced/h24/bigeval_live", "sync/h24"),
                  ("synced/h31", "sync/h31"), ("synced/h31/bigeval_live", "sync/h31"),
                  ("synced/h29", "sync/h29"), ("synced/h29/bigeval_live", "sync/h29")]:
    try:
        items = list_dir(path, ref)
        if isinstance(items, dict):
            out[f"{ref}:{path}"] = {"error": items}
        else:
            cal = [i for i in items if i["name"].endswith("_cal.out")]
            out[f"{ref}:{path}"] = {"count": len(items),
                                     "cal_outs": [i["name"] for i in cal]}
    except Exception as e:
        out[f"{ref}:{path}"] = {"error": str(e)[:200]}

print(json.dumps(out, indent=1, ensure_ascii=False))
