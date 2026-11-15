"""H23 status trip 06:42 - direct read of sync/h29 + sync/h23 branches.
Tag: H23STATUS0642. Read-only; prints branch heads, live-file line counts,
last-game times, *_cal.out contents, and latest Actions runs (channel check).
No writes, no dispatches."""
import base64, json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"

def b64decode(content):
    return base64.b64decode("".join(content.split())).decode("utf-8", errors="replace")

def get_text(branch, path):
    r = gh_api.get_file(OWNER, REPO, path, ref=branch)
    if isinstance(r, dict) and r.get("content"):
        return b64decode(r["content"])
    return None

def list_dir(branch, path):
    try:
        r = gh_api.get_file(OWNER, REPO, path, ref=branch)
    except Exception as e:
        return {"error": str(e)[:200]}
    if isinstance(r, list):
        return [(e["name"], e.get("size"), e.get("type")) for e in r]
    return {"error": f"unexpected: {str(r)[:200]}"}

out = {}
# 1. sync/h29 branch head
br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/sync/h29")
out["h29_head_sha"] = br["commit"]["sha"]
out["h29_head_date"] = br["commit"]["commit"]["author"]["date"]
out["h29_head_msg"] = br["commit"]["commit"]["message"][:120]

# 2. sync/h23 branch head
br23 = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/sync/h23")
out["h23_head_sha"] = br23["commit"]["sha"]
out["h23_head_date"] = br23["commit"]["commit"]["author"]["date"]
out["h23_head_msg"] = br23["commit"]["commit"]["message"][:120]

# 3. live dir listing + counts
live = list_dir("sync/h29", "synced/h29/bigeval_live")
out["live_listing"] = live
out["live_files"] = {}
total = 0
if isinstance(live, list):
    for name, size, _t in live:
        if not name.endswith(".jsonl"):
            continue
        txt = get_text("sync/h29", f"synced/h29/bigeval_live/{name}")
        if txt is None:
            out["live_files"][name] = {"error": "fetch failed/empty"}
            continue
        lines = [l for l in txt.splitlines() if l.strip()]
        last = None
        try:
            last = json.loads(lines[-1]) if lines else None
        except Exception:
            last = {"raw_tail": lines[-1][:200] if lines else None}
        out["live_files"][name] = {"lines": len(lines), "size": size, "last": last}
        total += len(lines)
out["total_rows"] = total

# 4. synced/h29 root listing (cal.out files)
root = list_dir("sync/h29", "synced/h29")
out["root_listing"] = root
out["cal_outs"] = {}
if isinstance(root, list):
    for name, size, _t in root:
        if name.endswith("_cal.out") or name.endswith(".out"):
            txt = get_text("sync/h29", f"synced/h29/{name}")
            out["cal_outs"][name] = {"size": size, "text": txt}

# 5. latest Actions runs (channel health)
runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs?per_page=5")
out["actions_latest"] = [
    {"id": r["id"], "name": r["name"], "status": r["status"],
     "conclusion": r["conclusion"], "created_at": r["created_at"],
     "head_branch": r.get("head_branch")}
    for r in runs.get("workflow_runs", [])
]

# 6. rate limit
rl = gh_api.req("GET", "/rate_limit")
out["rate_limit_core"] = rl["resources"]["core"]

print(json.dumps(out, indent=1, ensure_ascii=False))
