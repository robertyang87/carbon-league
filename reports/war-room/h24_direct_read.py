import json, sys
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

def branch(name):
    d = req("GET", f"/repos/robertyang87/carbon-league/branches/{name}")
    c = d["commit"]["commit"]
    return {"sha": d["commit"]["sha"][:8], "date": c["committer"]["date"], "msg": c["message"]}

def listing(path, ref):
    d = req("GET", f"/repos/robertyang87/carbon-league/contents/{path}?ref={ref}")
    if isinstance(d, dict):
        return {"error": d.get("message")}
    return [{"name": x["name"], "size": x.get("size"), "type": x["type"]} for x in d]

out = {}
out["sync/h24"] = branch("sync/h24")
out["sync/h31"] = branch("sync/h31")
out["synced_h24_root"] = listing("synced/h24", "sync/h24")
out["synced_h24_bigeval_live"] = listing("synced/h24/bigeval_live", "sync/h24")
out["synced_h31_bigeval_live"] = listing("synced/h31/bigeval_live", "sync/h31")
print(json.dumps(out, indent=1, ensure_ascii=False))
with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24-read-1644.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
