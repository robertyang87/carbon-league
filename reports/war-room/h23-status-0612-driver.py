"""H23STATUS0612: light direct read of sync/h23 + sync/h29 heads, rate limit,
Actions runs (unique marker H23STATUS0612). If sync/h29 head is unchanged
from 53f3c64f, git content-addressing makes the 8 live files byte-identical
to the 05:57 full direct count (664/1600) - no re-fetch needed.
"""
import sys, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {}
for name in ("sync/h23", "sync/h29"):
    try:
        b = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{name}")
        c = b["commit"]["commit"]
        out[name] = {"sha": b["commit"]["sha"][:7],
                     "date": c["committer"]["date"], "msg": c["message"][:100]}
    except Exception as e:
        out[name] = {"error": str(e)[:200]}
try:
    r = gh_api.req("GET", "/rate_limit")
    out["rate"] = r["resources"]["core"]
except Exception as e:
    out["rate"] = {"error": str(e)[:200]}
try:
    runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs?per_page=5")
    out["actions_runs"] = [
        {"id": x["id"], "name": x["name"], "status": x["status"],
         "conclusion": x.get("conclusion"), "created_at": x["created_at"],
         "head_branch": x.get("head_branch")}
        for x in runs.get("workflow_runs", [])]
except Exception as e:
    out["actions_runs"] = {"error": str(e)[:200]}
print(json.dumps(out, indent=1, ensure_ascii=False))
