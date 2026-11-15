"""H24 direct read probe, trip 2026-11-13 ~23:59 CST. Marker: h24-2359."""
import sys, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"

def branch_head(b):
    d = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{b}")
    c = d["commit"]["commit"]
    return {"branch": b, "sha": d["commit"]["sha"][:8],
            "date": c["committer"]["date"], "msg": c["message"][:80]}

def list_dir(path, ref):
    out, page = [], 1
    while True:
        d = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={ref}&per_page=100&page={page}")
        if not isinstance(d, list) or not d:
            break
        out.extend(d)
        if len(d) < 100:
            break
        page += 1
    return out

res = {}
for b in ["sync/h24", "sync/h31", "sync/h29", "main"]:
    try:
        res[b] = branch_head(b)
    except Exception as e:
        res[b] = {"error": str(e)[:200]}

# synced/h24 contents on sync/h24
try:
    items = list_dir("synced/h24", "sync/h24")
    outs = [i for i in items if i["name"].endswith(".out")]
    cals = [i for i in items if i["name"].endswith("_cal.out")]
    res["h24_synced"] = {"total": len(items), "out_count": len(outs),
                         "cal_outs": [c["name"] for c in cals]}
    # read the cal.out content
    for c in cals:
        f = gh_api.get_file(OWNER, REPO, "synced/h24/" + c["name"], ref="sync/h24")
        res["h24_cal_" + c["name"]] = f.get("content", "")[:1500] if isinstance(f, dict) else str(f)[:500]
except Exception as e:
    res["h24_synced"] = {"error": str(e)[:300]}

for b, sub in [("sync/h24", "synced/h24/bigeval_live"),
               ("sync/h31", "synced/h31/bigeval_live"),
               ("sync/h29", "synced/h29/bigeval_live")]:
    try:
        items = list_dir(sub, b)
        res[b + "_live"] = {"count": len(items),
                            "last": sorted([i["name"] for i in items])[-3:] if items else []}
    except Exception as e:
        res[b + "_live"] = {"error": str(e)[:300]}

print(json.dumps(res, ensure_ascii=False, indent=1))
