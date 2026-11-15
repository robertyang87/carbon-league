"""War-room-sync 12:22 trip (2026-11-14): branch heads, minimal API calls."""
import json, sys
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

OWNER, REPO = "robertyang87", "carbon-league"
OUT = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/war-room-sync-1303-branches.json"
res = {"read_at_cst": "2026-11-14 ~12:22 CST"}
for br in ["main", "sync/h29", "sync/h31", "sync/h23", "sync/h24", "sync/h28"]:
    try:
        b = req("GET", f"/repos/{OWNER}/{REPO}/branches/{br}")
        c = b["commit"]["commit"]
        res[br] = {"sha": b["commit"]["sha"][:8], "date": c["committer"]["date"],
                   "msg": c["message"].splitlines()[0][:100]}
    except Exception as e:
        res[br] = {"error": str(e)[:200]}
try:
    r = req("GET", "/rate_limit")
    res["rate_limit"] = {"core_remaining": r["resources"]["core"]["remaining"],
                         "core_reset": r["resources"]["core"]["reset"]}
except Exception as e:
    res["rate_limit"] = {"error": str(e)[:120]}
with open(OUT, "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
