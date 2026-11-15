"""H23 status probe 09:57 (marker H23STATUS1012).
Reads sync/h29 + sync/h23 branch heads, bigeval_live line counts,
cal.out files, latest workflow runs, rate limit. No writes, no machine commands.
"""
import base64, json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
MARK = "H23STATUS1012"

def branch_head(br):
    try:
        b = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{br}")
        c = b["commit"]["commit"]
        return {"branch": br, "sha": b["commit"]["sha"][:8],
                "committer_date": c["committer"]["date"],
                "message": c["message"][:120]}
    except Exception as e:
        return {"branch": br, "error": str(e)[:300]}

def list_dir(path, ref):
    try:
        r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={ref}")
        if isinstance(r, list):
            return [{"name": e["name"], "type": e["type"], "size": e.get("size")}
                    for e in r]
        return {"not_a_dir": str(r)[:200]}
    except Exception as e:
        return {"error": str(e)[:300]}

def get_text(path, ref):
    try:
        r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={ref}")
        if isinstance(r, dict) and r.get("encoding") == "base64":
            return base64.b64decode(r["content"]).decode("utf-8", errors="replace")
        return None
    except Exception as e:
        return f"<ERR {e}>"[:300]

out = {"marker": MARK}
out["h23_branch"] = branch_head("sync/h23")
out["h29_branch"] = branch_head("sync/h29")
out["h29_root"] = list_dir("synced/h29", "sync/h29")
out["h29_live_dir"] = list_dir("synced/h29/bigeval_live", "sync/h29")

live = out["h29_live_dir"]
if isinstance(live, list):
    out["live_files"] = []
    for e in live:
        if e["type"] != "file":
            continue
        txt = get_text(f"synced/h29/bigeval_live/{e['name']}", "sync/h29")
        if txt is None:
            out["live_files"].append({"name": e["name"], "error": "no content"})
            continue
        lines = [l for l in txt.splitlines() if l.strip()]
        out["live_files"].append({
            "name": e["name"], "size": e["size"], "lines": len(lines),
            "last": lines[-1][:300] if lines else ""})

root = out["h29_root"]
if isinstance(root, list):
    out["cal_files"] = []
    for e in root:
        if e["type"] == "file" and e["name"].endswith("_cal.out"):
            txt = get_text(f"synced/h29/{e['name']}", "sync/h29")
            out["cal_files"].append({"name": e["name"], "size": e["size"],
                                     "content": txt})

try:
    runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs?per_page=5")
    out["recent_runs"] = [{"id": r["id"], "name": r["name"],
                           "status": r["status"], "conclusion": r.get("conclusion"),
                           "created_at": r["created_at"]}
                          for r in runs.get("workflow_runs", [])]
except Exception as e:
    out["recent_runs"] = {"error": str(e)[:300]}

try:
    rl = gh_api.req("GET", "/rate_limit")
    out["rate_limit"] = rl["resources"]["core"]
except Exception as e:
    out["rate_limit"] = {"error": str(e)[:300]}

print(json.dumps(out, indent=2, ensure_ascii=False))
