import json, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

OUT = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24-read-0530.json"

def branch(name):
    d = req("GET", f"/repos/robertyang87/carbon-league/branches/{name}")
    c = d["commit"]["commit"]
    return {"sha": d["commit"]["sha"][:8], "full_sha": d["commit"]["sha"],
            "date": c["committer"]["date"], "msg": c["message"]}

def listing(path, ref):
    d = req("GET", f"/repos/robertyang87/carbon-league/contents/{path}?ref={ref}")
    if isinstance(d, dict):
        return {"error": d.get("message")}
    return [{"name": x["name"], "size": x.get("size"), "type": x["type"]} for x in d]

def blob_text(path, ref):
    d = req("GET", f"/repos/robertyang87/carbon-league/contents/{path}?ref={ref}")
    if isinstance(d, dict) and d.get("encoding") == "base64":
        return {"sha": d.get("sha"), "size": d.get("size"),
                "text": base64.b64decode(d["content"]).decode("utf-8", "replace")}
    return {"error": str(d)[:300]}

out = {}
out["read_at_cst"] = "2026-11-14 05:30 CST"
out["sync/h24"] = branch("sync/h24")
out["main"] = branch("main")
root = listing("synced/h24", "sync/h24")
out["synced_h24_root"] = root
out["synced_h24_root_n"] = len(root) if isinstance(root, list) else None
live = listing("synced/h24/bigeval_live", "sync/h24")
out["synced_h24_bigeval_live"] = live
out["bigeval_live_n"] = len(live) if isinstance(live, list) else None
out["cal_outs"] = [i["name"] for i in root if isinstance(i, dict) and i["name"].endswith("_cal.out")] if isinstance(root, list) else []
out["m3s2r1_cal"] = blob_text("synced/h24/m3s2r1_cal.out", "sync/h24")
# rate limit
rl = req("GET", "/rate_limit")
out["rate_core_remaining"] = rl["resources"]["core"]["remaining"]
print(json.dumps({k: v for k, v in out.items() if k not in ("synced_h24_bigeval_live",)}, indent=1, ensure_ascii=False))
print("bigeval_live_n:", out["bigeval_live_n"])
with open(OUT, "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
