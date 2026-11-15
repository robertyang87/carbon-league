"""H23 status check driver — run 12:42 CST 2026-11-15 (tag H23STATUS1242).
Minimal calls: head unchanged => tree unchanged => contents unchanged.
"""
import hashlib, json, sys, datetime
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
TAG = "H23STATUS1242"
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S CST")
out = {"run_tag": TAG, "run_cst": now}

def branch(name):
    try:
        b = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{name}")
        c = b["commit"]["commit"]
        return {"sha": b["commit"]["sha"][:8], "date_utc": c["committer"]["date"], "message": c["message"][:100]}
    except Exception as e:
        return {"error": str(e)}

out["sync/h23"] = branch("sync/h23")

def listing(path, ref):
    try:
        items = gh_api.get_file(OWNER, REPO, path, ref=ref)
        if isinstance(items, dict):
            return {"count": 1, "names": [items.get("name")], "error": items.get("message")}
        names = [i["name"] for i in items]
        blob = json.dumps(names, sort_keys=True).encode()
        return {"count": len(items), "md5": hashlib.md5(blob).hexdigest(), "tail_names": names[-5:], "error": None}
    except Exception as e:
        return {"count": None, "names": [], "error": str(e)}

out["synced_h23"] = listing("synced/h23", "sync/h23")
out["bigeval_live"] = listing("synced/h23/bigeval_live", "sync/h23")

try:
    rl = gh_api.req("GET", "/rate_limit")
    core = rl["resources"]["core"]
    out["api_core"] = {"remaining": core["remaining"], "reset_utc": core["reset"]}
except Exception as e:
    out["api_core"] = {"error": str(e)}

print(json.dumps(out, indent=1, ensure_ascii=False))
