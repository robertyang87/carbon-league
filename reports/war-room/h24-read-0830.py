"""H24 08:30 status read driver. Branch heads for sync/h24, sync/h30, sync/h29;
latest Actions runs; rate limit; newest run job detail if newer than 43335641497."""
import json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {"run_tag": "h24-0830", "checked_at_cst": "2026-11-13 08:32"}

for branch in ["sync/h24", "sync/h30", "sync/h29"]:
    try:
        commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={branch}&per_page=1")
        c = commits[0]
        out[branch] = {"sha": c["sha"][:8],
                       "date": c["commit"]["committer"]["date"],
                       "message": c["commit"]["message"].splitlines()[0][:120]}
    except Exception as e:
        out[branch] = {"error": str(e)}

try:
    runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs?per_page=8")
    out["runs"] = [{"id": r["id"], "name": r["name"],
                    "created_at": r["created_at"], "status": r["status"],
                    "conclusion": r["conclusion"]} for r in runs["workflow_runs"]]
    out["total_runs"] = runs["total_count"]
except Exception as e:
    out["runs_error"] = str(e)

try:
    rate = gh_api.req("GET", "/rate_limit")
    out["rate"] = {"core_remaining": rate["resources"]["core"]["remaining"],
                   "core_limit": rate["resources"]["core"]["limit"],
                   "core_reset": rate["resources"]["core"]["reset"]}
except Exception as e:
    out["rate_error"] = str(e)

# job detail for newest run if newer than last known 43335641497
newest = out.get("runs", [{}])[0]
if newest.get("id") and newest["id"] != 43335641497:
    try:
        jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{newest['id']}/jobs?per_page=5")
        js = jobs.get("jobs", [])
        out["latest_run_job"] = {
            "run_id": newest["id"], "run_name": newest["name"],
            "jobs": [{"id": j["id"], "started_at": j.get("started_at"),
                      "completed_at": j.get("completed_at"),
                      "steps": len(j.get("steps", [])),
                      "conclusion": j.get("conclusion")} for j in js]}
    except Exception as e:
        out["latest_run_job_error"] = str(e)
else:
    out["latest_run_job"] = {"note": "no run newer than 43335641497"}

print(json.dumps(out, indent=1, ensure_ascii=False))
with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24-read-0830.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
