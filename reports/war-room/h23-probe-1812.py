"""H23 status probe 18:12 - read sync/h23 branch via GitHub API."""
import json, sys, base64, urllib.request
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h23"

def get_json(path):
    return gh_api.req("GET", path)

def fetch_raw(path, ref):
    """Fetch a file's raw content via contents API (base64)."""
    d = gh_api.get_file(OWNER, REPO, path, ref=ref)
    if isinstance(d, dict) and d.get("content"):
        return base64.b64decode(d["content"]).decode("utf-8", errors="replace")
    return None

out = {}

# 1. latest commit on sync/h23
commits = get_json(f"/repos/{OWNER}/{REPO}/commits?sha={BR}&per_page=1")
c = commits[0]
out["latest_commit"] = {
    "sha": c["sha"][:7],
    "date_utc": c["commit"]["committer"]["date"],
    "message": c["commit"]["message"][:120],
}
print("LATEST COMMIT:", json.dumps(out["latest_commit"], ensure_ascii=False))

# 2. list synced/h23/ and synced/h23/bigeval_live/
for d in ["synced/h23", "synced/h23/bigeval_live"]:
    items = gh_api.get_file(OWNER, REPO, d, ref=BR)
    if isinstance(items, list):
        print(f"\nDIR {d} ({len(items)} items):")
        for it in items:
            print(f"  {it['name']}  size={it.get('size')}  type={it['type']}")
    else:
        print(f"\nDIR {d}: {json.dumps(items, ensure_ascii=False)[:300]}")
