"""H23 status check driver — run 03:12 CST 2026-11-14 (tag H23CHK0312)."""
import base64, json, sys, datetime
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {"run_tag": "H23CHK0312", "run_cst": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S CST")}

def branch(name):
    try:
        b = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{name}")
        c = b["commit"]["commit"]
        return {"sha": b["commit"]["sha"][:8], "date_utc": c["committer"]["date"], "message": c["message"][:80]}
    except Exception as e:
        return {"error": str(e)}

for name in ["sync/h23", "sync/h29", "sync/h31", "main"]:
    out[name] = branch(name)

def listing(path, ref):
    try:
        items = gh_api.get_file(OWNER, REPO, path, ref=ref)
        if isinstance(items, dict):
            return {"count": 1, "names": [items.get("name")], "error": items.get("message")}
        return {"count": len(items), "names": [i["name"] for i in items], "error": None}
    except Exception as e:
        return {"count": None, "names": [], "error": str(e)}

s23 = listing("synced/h23", "sync/h23")
out["synced_h23"] = {"count": s23["count"], "names": s23["names"][:60], "error": s23["error"]}
live = listing("synced/h23/bigeval_live", "sync/h23")
out["bigeval_live"] = {"count": live["count"], "error": live["error"],
                       "tail_names": live["names"][-5:] if live["names"] else []}

def calfile(name):
    try:
        f = gh_api.get_file(OWNER, REPO, f"synced/h23/{name}", ref="sync/h23")
        content = base64.b64decode(f["content"]).decode("utf-8", errors="replace") if f.get("content") else ""
        return {"size": f.get("size"), "content": content}
    except Exception as e:
        return {"size": None, "content": "", "error": str(e)}

out["cal_out"] = {n: calfile(n) for n in ["lineA_cal.out", "m3s2r1pre_cal.out", "m7bsr1_cal.out"]}

try:
    rl = gh_api.req("GET", "/rate_limit")
    core = rl["resources"]["core"]
    out["api_core"] = {"remaining": core["remaining"], "reset_utc": core["reset"]}
except Exception as e:
    out["api_core"] = {"error": str(e)}

print(json.dumps(out, indent=1, ensure_ascii=False))
