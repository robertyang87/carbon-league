"""H24 08:44 read driver: branch heads + latest Actions runs + rate limit."""
import json, sys
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

out = {"run_tag": "h24-0844", "checked_at_cst": "2026-11-13 08:47"}
for br in ("sync/h24", "sync/h30", "sync/h29"):
    b = req("GET", f"/repos/robertyang87/carbon-league/branches/{br}")
    c = b["commit"]["commit"]
    out[br] = {"sha": b["commit"]["sha"][:8], "date": c["committer"]["date"],
               "message": c["message"]}

runs = req("GET", "/repos/robertyang87/carbon-league/actions/runs?per_page=10")
out["runs"] = [{"id": r["id"], "name": r["name"], "created_at": r["created_at"],
                "status": r["status"], "conclusion": r["conclusion"]}
               for r in runs["workflow_runs"]]
out["total_runs"] = runs["total_count"]

rl = req("GET", "/rate_limit")
out["rate"] = {"core_remaining": rl["resources"]["core"]["remaining"],
               "core_limit": rl["resources"]["core"]["limit"],
               "reset_utc": rl["resources"]["core"]["reset"]}

# job detail for the newest run (same-signature check)
newest = runs["workflow_runs"][0]
jobs = req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{newest['id']}/jobs?per_page=5")
out["newest_run_jobs"] = [{"id": j["id"], "name": j["name"], "status": j["status"],
                           "conclusion": j["conclusion"], "started_at": j["started_at"],
                           "completed_at": j["completed_at"], "steps": len(j.get("steps", [])),
                           "runner_name": j.get("runner_name")}
                          for j in jobs["jobs"]]

with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24-read-0844.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2, ensure_ascii=False))
