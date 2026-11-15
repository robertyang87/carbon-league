"""h24 status read: sync/h24 + sync/h31 branch heads, root + bigeval_live listings."""
import json, sys
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

OUT = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24-read-1744.json"

def branch(name):
    b = req("GET", f"/repos/robertyang87/carbon-league/branches/{name}")
    c = b["commit"]["commit"]
    return {"sha": b["commit"]["sha"][:8], "date": c["committer"]["date"], "msg": c["message"]}

def listing(path, ref):
    items = req("GET", f"/repos/robertyang87/carbon-league/contents/{path}?ref={ref}")
    if isinstance(items, dict):
        return items
    return [{"name": i["name"], "size": i["size"], "type": i["type"]} for i in items]

res = {}
res["read_at_cst"] = "2026-11-13 17:44-17:45 CST"
res["sync/h24"] = branch("sync/h24")
res["sync/h31"] = branch("sync/h31")
res["synced_h24_root"] = listing("synced/h24", "sync/h24")
res["synced_h24_bigeval_live"] = listing("synced/h24/bigeval_live", "sync/h24")
res["h31_bigeval_live"] = listing("synced/h31/bigeval_live", "sync/h31")
hak = {i["name"]: i["size"] for i in res["h31_bigeval_live"]
       if isinstance(i, dict) and i["name"].startswith("hakimi3")}
res["hakimi_files"] = hak
res["h31_live_names"] = [i["name"] for i in res["h31_bigeval_live"] if isinstance(i, dict)]
res["h24_cal_outs"] = [i["name"] for i in res["synced_h24_root"]
                       if isinstance(i, dict) and i["name"].endswith("_cal.out")]
with open(OUT, "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print(json.dumps({"h24": res["sync/h24"], "h31": res["sync/h31"],
                  "h24_root_n": len(res["synced_h24_root"]),
                  "h24_live": res["synced_h24_bigeval_live"],
                  "h31_live_n": len(res["h31_bigeval_live"]),
                  "hakimi": hak, "cal_outs": res["h24_cal_outs"]}, ensure_ascii=False))
