"""H24 11:14 trip read. Branch head sync/h24 + main head + rate.
Head unchanged since last trip => live/cal contents unchanged, skip re-read."""
import json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {}
for br in ["sync/h24", "main"]:
    c = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={br}&per_page=1")
    out[br] = {"sha": c[0]["sha"][:8], "date": c[0]["commit"]["committer"]["date"],
               "msg": c[0]["commit"]["message"].splitlines()[0][:120]}
rl = gh_api.req("GET", "/rate_limit")
core = rl["resources"]["core"]
out["rate"] = {"core_remaining": core["remaining"], "core_limit": core["limit"],
               "core_reset": core["reset"]}
print(json.dumps(out, indent=1, ensure_ascii=False))
with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24-read-1114.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
