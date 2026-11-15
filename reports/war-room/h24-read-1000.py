"""H24 10:00 status read (2026-11-14): sync/h24 head, synced/h24 root +
bigeval_live listings, m3s2r1_cal.out full text, rate limit. Minimal calls."""
import json, sys
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req, get_file

OUT = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24-read-1000.json"
OWNER, REPO = "robertyang87", "carbon-league"
res = {"read_at_cst": "2026-11-14 ~10:00 CST"}

try:
    b = req("GET", f"/repos/{OWNER}/{REPO}/branches/sync/h24")
    c = b["commit"]["commit"]
    res["sync/h24"] = {"sha": b["commit"]["sha"], "date": c["committer"]["date"],
                       "msg": c["message"].splitlines()[0][:120]}
except Exception as e:
    res["sync/h24"] = {"error": str(e)}

def listing(path, ref):
    items = req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={ref}")
    if isinstance(items, dict):
        return items
    return [{"name": i["name"], "size": i["size"], "type": i["type"]} for i in items]

try:
    root = listing("synced/h24", "sync/h24")
    res["synced_h24_root_n"] = len(root)
    res["synced_h24_root"] = root if isinstance(root, dict) else root
    res["h24_cal_outs"] = [i["name"] for i in root
                           if isinstance(i, dict) and i["name"].endswith("_cal.out")]
except Exception as e:
    res["synced_h24_root_error"] = str(e)

try:
    live = listing("synced/h24/bigeval_live", "sync/h24")
    res["bigeval_live_n"] = len(live) if isinstance(live, list) else live
    if isinstance(live, list):
        # newest by name order is not reliable; keep last 5 names + sizes
        res["bigeval_live_tail"] = live[-5:]
except Exception as e:
    res["bigeval_live_error"] = str(e)

try:
    cal = get_file(OWNER, REPO, "synced/h24/m3s2r1_cal.out", "sync/h24")
    res["m3s2r1_cal.out"] = cal
except Exception as e:
    res["m3s2r1_cal.out_error"] = str(e)

try:
    rate = req("GET", "/rate_limit")
    res["rate"] = {"core_remaining": rate["resources"]["core"]["remaining"],
                   "core_limit": rate["resources"]["core"]["limit"]}
except Exception as e:
    res["rate_error"] = str(e)

with open(OUT, "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False)[:6000])
