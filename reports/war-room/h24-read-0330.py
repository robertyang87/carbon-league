import json, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

OUT = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24-read-0330.json"

def branch_head(branch):
    b = req("GET", f"/repos/robertyang87/carbon-league/branches/{branch}")
    c = b["commit"]["commit"]
    return {"sha": b["commit"]["sha"][:7], "date": c["committer"]["date"], "message": c["message"]}

res = {}
res["trip"] = "h24-status-report 03:30"
res["sync_h24_head"] = branch_head("sync/h24")
res["main_head"] = branch_head("main")
res["sync_h31_head"] = branch_head("sync/h31")

tree = req("GET", f"/repos/robertyang87/carbon-league/git/trees/{res['sync_h24_head']['sha']}?recursive=1")
paths = [t["path"] for t in tree.get("tree", [])]
h24 = [p for p in paths if p.startswith("synced/h24/")]
live = [p for p in h24 if p.startswith("synced/h24/bigeval_live/")]
res["synced_h24_items"] = len(h24)
res["bigeval_live_items"] = len(live)
res["cal_out_files"] = [p for p in h24 if p.endswith("_cal.out")]

cal = {}
for p in res["cal_out_files"]:
    f = req("GET", f"/repos/robertyang87/carbon-league/contents/{p}?ref=sync/h24")
    cal[p] = {"sha": f["sha"][:12], "size": f["size"],
              "text": base64.b64decode(f["content"]).decode()}
res["cal_out"] = cal

rl = req("GET", "/rate_limit")
res["rate_limit"] = {"limit": rl["resources"]["core"]["limit"],
                     "used": rl["resources"]["core"]["used"],
                     "remaining": rl["resources"]["core"]["remaining"],
                     "reset": rl["resources"]["core"]["reset"]}

with open(OUT, "w") as fh:
    json.dump(res, fh, indent=2, ensure_ascii=False)
print(json.dumps(res, indent=2, ensure_ascii=False))
