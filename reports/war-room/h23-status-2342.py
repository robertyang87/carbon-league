"""H23 status probe 22:27 - branch heads + synced/h23 counts + cal.out sizes + rate limit. Marker H23STAT2342."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

O, R = "robertyang87", "carbon-league"

def branch_head(br):
    try:
        b = gh_api.req("GET", f"/repos/{O}/{R}/branches/{br}")
        c = b["commit"]
        return {"branch": br, "sha": c["sha"][:8],
                "date": c["commit"]["committer"]["date"],
                "msg": c["commit"]["message"].split("\n")[0]}
    except Exception as e:
        return {"branch": br, "error": str(e)[:300]}

out = {}
out["h23_branch"] = branch_head("sync/h23")
out["h29_branch"] = branch_head("sync/h29")
out["main_branch"] = branch_head("main")

def dir_count(path, ref):
    try:
        items = gh_api.get_file(O, R, path, ref=ref)
        if isinstance(items, list):
            return {"path": path, "items": len(items)}
        return {"path": path, "type": type(items).__name__}
    except Exception as e:
        return {"path": path, "error": str(e)[:300]}

out["synced_h23"] = dir_count("synced/h23", "sync/h23")
out["bigeval_live_h23"] = dir_count("synced/h23/bigeval_live", "sync/h23")

def file_size(path, ref):
    try:
        f = gh_api.get_file(O, R, path, ref=ref)
        raw = base64.b64decode(f["content"]).decode(errors="replace")
        return {"path": path, "bytes": len(raw.encode()), "tail": raw.strip().splitlines()[-1][:200] if raw.strip() else ""}
    except Exception as e:
        return {"path": path, "error": str(e)[:300]}

for name in ["m3s2r1pre_cal.out", "lineA_cal.out", "m7bsr1_cal.out"]:
    out[name] = file_size(f"synced/h23/{name}", "sync/h23")

try:
    rl = gh_api.req("GET", "/rate_limit")
    core = rl["resources"]["core"]
    out["rate_limit"] = {"remaining": core["remaining"], "reset": core["reset"]}
except Exception as e:
    out["rate_limit"] = {"error": str(e)[:200]}

print(json.dumps(out, ensure_ascii=False, indent=1))
with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h23-status-2342.log", "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
