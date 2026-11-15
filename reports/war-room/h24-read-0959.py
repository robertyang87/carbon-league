"""H24 0959 trip direct read: branch heads (h24/h29/h30), latest cloud-assistant runs, rate limit.
No file re-reads: heads unchanged since 09:30 means cal.out/live contents are already known."""
import json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {}
for b in ["sync/h24", "sync/h29", "sync/h30"]:
    c = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={b}&per_page=1")
    out[b] = {"sha": c[0]["sha"][:8],
              "date": c[0]["commit"]["committer"]["date"],
              "msg": c[0]["commit"]["message"].splitlines()[0][:120]}
runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs?per_page=5")
out["runs"] = [{"id": r["id"], "name": r["name"], "created_at": r["created_at"],
                "status": r["status"], "conclusion": r["conclusion"]}
               for r in runs["workflow_runs"]]
out["total_runs"] = runs["total_count"]
rl = gh_api.req("GET", "/rate_limit")
out["rate"] = {"core_remaining": rl["resources"]["core"]["remaining"],
               "core_limit": rl["resources"]["core"]["limit"],
               "core_reset": rl["resources"]["core"]["reset"]}
print(json.dumps(out, indent=1, ensure_ascii=False))
with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24-read-0959.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
