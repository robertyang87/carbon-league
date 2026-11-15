"""H23 status probe 10:27b (marker H23STATUS1027B). Lean, incremental output.
Step 1: branch heads + recent runs + rate limit.
Step 2: sync/h29 root + live dir + per-file line counts.
Step 3: sync/h23 root + live dir listing only (sizes as proxy).
Each step flushed to the log file immediately.
"""
import base64, json, sys, time
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
LOG = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h23-probe-1027b.log"

def log(obj):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()

def branch_head(br):
    t0 = time.time()
    try:
        b = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{br}")
        c = b["commit"]["commit"]
        return {"branch": br, "sha": b["commit"]["sha"][:8],
                "committer_date": c["committer"]["date"],
                "message": c["message"][:120], "secs": round(time.time() - t0, 1)}
    except Exception as e:
        return {"branch": br, "error": str(e)[:300], "secs": round(time.time() - t0, 1)}

def list_dir(path, ref):
    t0 = time.time()
    try:
        r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={ref}")
        if isinstance(r, list):
            return {"path": path, "entries": [{"name": e["name"], "type": e["type"], "size": e.get("size")} for e in r], "secs": round(time.time() - t0, 1)}
        return {"path": path, "not_a_dir": str(r)[:200]}
    except Exception as e:
        return {"path": path, "error": str(e)[:300], "secs": round(time.time() - t0, 1)}

def get_text(path, ref):
    t0 = time.time()
    try:
        r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={ref}")
        if isinstance(r, dict) and r.get("encoding") == "base64":
            return base64.b64decode(r["content"]).decode("utf-8", errors="replace"), round(time.time() - t0, 1)
        return None, round(time.time() - t0, 1)
    except Exception as e:
        return f"<ERR {e}>"[:300], round(time.time() - t0, 1)

with open(LOG, "w", encoding="utf-8") as f:
    f.write('{"marker": "H23STATUS1027B", "started": "10:27b"}\n')

log({"step": "heads", "h23": branch_head("sync/h23"), "h29": branch_head("sync/h29")})

try:
    runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs?per_page=5")
    log({"step": "runs", "recent_runs": [{"id": r["id"], "name": r["name"], "status": r["status"], "conclusion": r.get("conclusion"), "created_at": r["created_at"]} for r in runs.get("workflow_runs", [])]})
except Exception as e:
    log({"step": "runs", "error": str(e)[:300]})

try:
    rl = gh_api.req("GET", "/rate_limit")
    log({"step": "rate_limit", "core": rl["resources"]["core"]})
except Exception as e:
    log({"step": "rate_limit", "error": str(e)[:300]})

root = list_dir("synced/h29", "sync/h29")
log({"step": "h29_root", **root})
live = list_dir("synced/h29/bigeval_live", "sync/h29")
log({"step": "h29_live_dir", **live})
if isinstance(live, dict) and "entries" in live:
    files = []
    for e in live["entries"]:
        if e["type"] != "file":
            continue
        txt, secs = get_text(f"synced/h29/bigeval_live/{e['name']}", "sync/h29")
        if txt is None:
            files.append({"name": e["name"], "error": "no content"})
            continue
        lines = [l for l in txt.splitlines() if l.strip()]
        files.append({"name": e["name"], "size": e["size"], "lines": len(lines),
                      "last": lines[-1][:200] if lines else "", "secs": secs})
    log({"step": "h29_live_files", "files": files, "total_lines": sum(f.get("lines", 0) for f in files)})
    if isinstance(root, dict) and "entries" in root:
        cals = []
        for e in root["entries"]:
            if e["type"] == "file" and e["name"].endswith("_cal.out"):
                txt, secs = get_text(f"synced/h29/{e['name']}", "sync/h29")
                cals.append({"name": e["name"], "size": e["size"], "content": txt})
        log({"step": "h29_cal_files", "cal_files": cals})

log({"step": "h23_root", **list_dir("synced/h23", "sync/h23")})
log({"step": "h23_live_dir", **list_dir("synced/h23/bigeval_live", "sync/h23")})
log({"step": "done"})
print("DONE")
