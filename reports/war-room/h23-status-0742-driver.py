"""H23 status-report 07:42 trip driver. One-shot read: sync/h29 + sync/h23 branch
heads, latest Actions runs, rate limit. No dispatch, no retry loop."""
import json, sys, datetime
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

def cst(iso):
    dt = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return dt.astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")

out = {}
for br in ("sync/h29", "sync/h23"):
    b = req("GET", f"/repos/robertyang87/carbon-league/branches/{br}")
    c = b["commit"]["commit"]
    out[br] = {"sha": b["commit"]["sha"][:8], "date_cst": cst(c["committer"]["date"]),
               "msg": c["message"].splitlines()[0][:80]}
runs = req("GET", "/repos/robertyang87/carbon-league/actions/runs?per_page=5")
out["actions_latest"] = [
    {"id": r["id"], "name": r["name"], "created_cst": cst(r["created_at"]),
     "status": r["status"], "conclusion": r["conclusion"]}
    for r in runs["workflow_runs"]]
rl = req("GET", "/rate_limit")
core = rl["resources"]["core"]
out["rate_limit"] = {"remaining": core["remaining"],
                     "reset_cst": cst(datetime.datetime.fromtimestamp(core["reset"], datetime.timezone.utc).isoformat())}
out["checked_at_cst"] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
print(json.dumps(out, ensure_ascii=False, indent=1))
