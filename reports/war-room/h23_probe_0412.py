import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

R = "robertyang87/carbon-league"
out = {}

def branch_head(b):
    try:
        d = gh_api.req("GET", f"/repos/{R}/branches/{b}")
        return {"sha": d["commit"]["sha"][:8], "date": d["commit"]["commit"]["committer"]["date"]}
    except Exception as e:
        return {"error": str(e)}

for b in ["sync/h23", "sync/h29", "sync/h31", "main"]:
    out["branch_" + b.replace("/", "_")] = branch_head(b)

def listing(path, ref):
    try:
        d = gh_api.get_file("robertyang87", "carbon-league", path, ref=ref)
        if isinstance(d, list):
            return {"count": len(d), "names": [x["name"] for x in d][:50]}
        return {"type": type(d).__name__, "keys": list(d.keys())[:10]}
    except Exception as e:
        return {"error": str(e)}

out["synced_h23"] = listing("synced/h23", "sync/h23")
out["bigeval_live"] = listing("synced/h23/bigeval_live", "sync/h23")

# cal.out files: find them in synced/h23 listing
try:
    d = gh_api.get_file("robertyang87", "carbon-league", "synced/h23", ref="sync/h23")
    cals = [x["name"] for x in d if x["name"].endswith("cal.out")]
    out["cal_files"] = cals
    for name in cals:
        try:
            f = gh_api.get_file("robertyang87", "carbon-league", f"synced/h23/{name}", ref="sync/h23")
            content = base64.b64decode(f["content"]).decode("utf-8", "replace")
            out["cal_" + name] = {"size": f["size"], "content": content}
        except Exception as e:
            out["cal_" + name] = {"error": str(e)}
except Exception as e:
    out["cal_list_error"] = str(e)

try:
    rl = gh_api.req("GET", "/rate_limit")
    out["rate_limit"] = {"core_remaining": rl["resources"]["core"]["remaining"],
                         "core_reset_utc": rl["resources"]["core"]["reset"]}
except Exception as e:
    out["rate_limit"] = {"error": str(e)}

print(json.dumps(out, ensure_ascii=False, indent=1))
