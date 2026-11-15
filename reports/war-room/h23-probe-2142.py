"""H23 sync/h23 branch probe — run 21:42, marker H23STATUS2142."""
import base64, json, sys
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req, get_file

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h23"

def list_dir(path):
    return req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={BRANCH}")

def file_bytes(path):
    d = get_file(OWNER, REPO, path, ref=BRANCH)
    if isinstance(d, dict) and d.get("content"):
        return base64.b64decode(d["content"])
    return None

out = {}
# 1. latest commits on sync/h23
commits = req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=3")
out["latest_commits"] = [
    {"sha": c["sha"][:7], "date": c["commit"]["committer"]["date"],
     "msg": c["commit"]["message"][:80]} for c in commits]
# 2. directory listings
for path in ["synced/h23", "synced/h23/bigeval_live"]:
    try:
        items = list_dir(path)
        out[path] = [{"name": i["name"], "size": i["size"], "type": i["type"]}
                     for i in items] if isinstance(items, list) else {"error": str(items)[:200]}
    except Exception as e:
        out[path] = {"error": str(e)[:300]}
# 3. key files: line counts / sizes / tail
for path in ["synced/h23/bigeval_live/k4v2r1_e0.jsonl",
             "synced/h23/bigeval_live/k4probe.out",
             "synced/h23/bigeval_live/k4v2r1_e0.launcher.log",
             "synced/h23/m7bsr1_cal.out",
             "synced/h23/bigeval_live/m7bsr1_m2.jsonl",
             "synced/h23/bigeval_live/m7bsr1_echo2.jsonl"]:
    try:
        b = file_bytes(path)
        if b is None:
            out[path] = {"status": "no-content-or-missing"}
        else:
            lines = b.decode("utf-8", errors="replace").splitlines()
            out[path] = {"bytes": len(b), "lines": len(lines),
                         "last_line": lines[-1][:200] if lines else None}
    except Exception as e:
        out[path] = {"error": str(e)[:300]}

print(json.dumps(out, indent=1, ensure_ascii=False))
