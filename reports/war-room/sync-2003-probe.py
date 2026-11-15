"""sync trip 20:03 probe: batch-read main + 5 sync branch heads + rate limit."""
import sys, json
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

OWNER, REPO = "robertyang87", "carbon-league"
out = {"probe": "sync-2003"}
for br in ["main", "sync/h23", "sync/h24", "sync/h28", "sync/h29", "sync/h31"]:
    b = req("GET", f"/repos/{OWNER}/{REPO}/branches/{br}")
    c = b["commit"]["commit"]
    out[br] = {
        "sha": b["commit"]["sha"][:8],
        "time_utc": c["committer"]["date"],
        "msg": c["message"][:100],
    }
rl = req("GET", "/rate_limit")
out["api_core_remaining"] = rl["resources"]["core"]["remaining"]
out["api_core_reset_utc"] = rl["resources"]["core"]["reset"]
print(json.dumps(out, indent=2, ensure_ascii=False))
