"""H24 direct read probe, trip 2026-11-15 ~00:14 CST. Marker: h24-0014."""
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
try:
    res["sync_h24_head"] = branch_head("sync/h24")
except Exception as e:
    res["sync_h24_head"] = {"error": str(e)[:200]}

try:
    items = list_dir("synced/h24", "sync/h24")
    outs = [i for i in items if i["name"].endswith(".out")]
    cals = [i for i in items if i["name"].endswith("_cal.out")]
    res["h24_synced"] = {"total": len(items), "out_count": len(outs),
                         "cal_outs": [c["name"] for c in cals]}
    for c in cals:
        f = gh_api.get_file(OWNER, REPO, "synced/h24/" + c["name"], ref="sync/h24")
        if isinstance(f, dict) and f.get("encoding") == "base64":
            import base64
            res["h24_cal_" + c["name"]] = base64.b64decode(f["content"]).decode(errors="replace")[:1500]
        else:
            res["h24_cal_" + c["name"]] = str(f)[:500]
        res["h24_cal_meta_" + c["name"]] = {"size": c.get("size"), "sha": (c.get("sha") or "")[:8]}
except Exception as e:
    res["h24_synced"] = {"error": str(e)[:300]}

try:
    items = list_dir("synced/h24/bigeval_live", "sync/h24")
    res["h24_live"] = {"count": len(items)}
    # the three live files from previous trips: match by name substring
    for key in ("m2", "e0", "echo2"):
        hits = [i for i in items if key in i["name"].lower()]
        res["h24_live_" + key] = [{"name": i["name"], "size": i.get("size"),
                                   "sha": (i.get("sha") or "")[:8]} for i in hits]
    res["h24_live_last3"] = sorted([i["name"] for i in items])[-3:]
except Exception as e:
    res["h24_live"] = {"error": str(e)[:300]}

try:
    rl = gh_api.req("GET", "/rate_limit")
    res["api_core_remaining"] = rl["resources"]["core"]["remaining"]
except Exception as e:
    res["api_core_remaining"] = "error: " + str(e)[:100]

print(json.dumps(res, ensure_ascii=False, indent=1))
