"""H24 status check: sync branch heads + latest cloud-assistant runs + rate limit.
One batched read per run; no dispatch, no writes."""
import sys, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"

def branch_head(b):
    try:
        r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{b}")
        c = r["commit"]["commit"]
        return {"branch": b, "sha": r["commit"]["sha"][:8],
                "date": c["committer"]["date"], "msg": c["message"][:80]}
    except Exception as e:
        return {"branch": b, "error": str(e)[:200]}

out = {}
out["rate_limit"] = gh_api.req("GET", "/rate_limit")["resources"]["core"]
for b in ["sync/h24", "sync/h29", "sync/h30"]:
    out[b] = branch_head(b)
# latest workflow runs (all workflows, newest first) - identifies channel state
try:
    runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs?per_page=5")
    out["recent_runs"] = [
        {"id": r["id"], "name": r["name"], "status": r["status"],
         "conclusion": r["conclusion"], "created_at": r["created_at"],
         "head_branch": r["head_branch"]}
        for r in runs["workflow_runs"]
    ]
except Exception as e:
    out["recent_runs_error"] = str(e)[:200]
# bigeval_live listing on sync/h24 (stale expected; machine released)
try:
    d = gh_api.get_file(OWNER, REPO, "synced/h24/bigeval_live", ref="sync/h24")
    out["h24_bigeval_live"] = [x["name"] for x in d] if isinstance(d, list) else d.get("message", str(d)[:200])
except Exception as e:
    out["h24_bigeval_live_error"] = str(e)[:200]
print(json.dumps(out, indent=1, ensure_ascii=False))
