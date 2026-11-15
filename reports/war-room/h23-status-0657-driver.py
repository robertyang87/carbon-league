"""H23 status trip 06:57 - lean direct read of sync/h29 + sync/h23 branch heads,
latest Actions runs (channel check), and rate limit.
Tag: H23STATUS0657. Read-only; no writes, no dispatches.
Economy rule: live-file contents are versioned with the branch head; if the
h29 head SHA is unchanged since the last full count (53f3c64f @04:12:54, counts
48/51/114/121/111/55/56/108 = 664/1600), the counts stand without re-fetching
all eight ~400KB files. Only fetch files if the head moved."""
import json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {}

br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/sync/h29")
out["h29_head_sha"] = br["commit"]["sha"]
out["h29_head_date"] = br["commit"]["commit"]["author"]["date"]
out["h29_head_msg"] = br["commit"]["commit"]["message"][:120]

br23 = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/sync/h23")
out["h23_head_sha"] = br23["commit"]["sha"]
out["h23_head_date"] = br23["commit"]["commit"]["author"]["date"]
out["h23_head_msg"] = br23["commit"]["commit"]["message"][:120]

runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs?per_page=5")
out["actions_latest"] = [
    {"id": r["id"], "name": r["name"], "status": r["status"],
     "conclusion": r["conclusion"], "created_at": r["created_at"],
     "head_branch": r.get("head_branch")}
    for r in runs.get("workflow_runs", [])
]

rl = gh_api.req("GET", "/rate_limit")
out["rate_limit_core"] = rl["resources"]["core"]

print(json.dumps(out, indent=1, ensure_ascii=False))
