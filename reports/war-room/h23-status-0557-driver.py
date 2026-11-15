"""H23STATUS0557: direct read of sync/h23 + sync/h29 heads, rate limit,
Actions runs, and sync/h29 bigeval_live line counts (unique marker H23STATUS0557)."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {}
for name in ("sync/h23", "sync/h29"):
    try:
        b = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{name}")
        c = b["commit"]["commit"]
        out[name] = {"sha": b["commit"]["sha"][:7],
                     "date": c["committer"]["date"], "msg": c["message"][:100]}
    except Exception as e:
        out[name] = {"error": str(e)[:200]}
try:
    r = gh_api.req("GET", "/rate_limit")
    out["rate"] = r["resources"]["core"]
except Exception as e:
    out["rate"] = {"error": str(e)[:200]}
try:
    runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs?per_page=5")
    out["actions_runs"] = [
        {"id": x["id"], "name": x["name"], "status": x["status"],
         "conclusion": x.get("conclusion"), "created_at": x["created_at"],
         "head_branch": x.get("head_branch")}
        for x in runs.get("workflow_runs", [])]
except Exception as e:
    out["actions_runs"] = {"error": str(e)[:200]}
# live files under synced/h29/bigeval_live/ : direct line counts
try:
    tree = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/git/trees/"
                             f"{out['sync/h29']['sha']}?recursive=1")
    live = [t for t in tree.get("tree", [])
            if t["path"].startswith("synced/h29/bigeval_live/")
            and t["type"] == "blob"]
    out["live_files"] = []
    total = 0
    for t in sorted(live, key=lambda x: x["path"]):
        try:
            f = gh_api.get_file(OWNER, REPO, t["path"], ref="sync/h29")
            raw = base64.b64decode(f["content"])
            lines = raw.decode("utf-8", "replace").splitlines()
            n = len([l for l in lines if l.strip()])
            total += n
            last = lines[-1][:160] if lines else ""
            out["live_files"].append({"path": t["path"].split("/")[-1],
                                      "sha": t["sha"][:7], "size": t["size"],
                                      "lines": n, "last_line_head": last})
        except Exception as e:
            out["live_files"].append({"path": t["path"], "error": str(e)[:200]})
    out["backfill_total"] = total
except Exception as e:
    out["live_files_error"] = str(e)[:300]
print(json.dumps(out, indent=1, ensure_ascii=False))
