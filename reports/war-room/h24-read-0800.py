"""H24 lean direct read, run 2026-11-13 ~08:00 CST.
Cheap checks only: rate limit, branch heads (h24/h30/h29), newest
workflow runs (channel evidence). Full cal.out/live re-read only if a
head moved since the last verified read."""
import json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {}

rl = gh_api.req("GET", "/rate_limit")
out["rate_limit"] = rl["resources"]["core"]

def head(branch):
    cs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={branch}&per_page=2")
    return [{"sha": c["sha"][:7], "date": c["commit"]["committer"]["date"],
             "msg": c["commit"]["message"].splitlines()[0][:120]} for c in cs]

for b in ["sync/h24", "sync/h30", "sync/h29"]:
    out[b] = head(b)

runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs?per_page=12")
out["recent_runs"] = [
    {"id": r["id"], "name": r["name"], "created_at": r["created_at"],
     "status": r["status"], "conclusion": r["conclusion"],
     "head_branch": r["head_branch"]}
    for r in runs["workflow_runs"]
]
print(json.dumps(out, indent=1, ensure_ascii=False))
