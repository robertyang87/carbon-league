"""H24 status probe for 2026-11-12 20:29 run. Marker: H24PROBE2029."""
import json, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h24"

def show(label, obj):
    print(f"=== {label} ===")
    print(json.dumps(obj, indent=1, ensure_ascii=False)[:6000])

# 1. Latest commits on sync/h24
commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=5")
for c in commits:
    print("COMMIT", c["sha"][:7], c["commit"]["committer"]["date"], c["commit"]["message"].splitlines()[0] if c["commit"]["message"] else "")

# 2. Tree of synced/h24 (recursive, top-level entries + bigeval_live)
tree = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/git/trees/{BRANCH}?recursive=1")
paths = [t["path"] for t in tree.get("tree", []) if t["path"].startswith("synced/h24/")]
for p in sorted(paths):
    print("PATH", p)

# 3. Live files: line counts via raw content
def get_raw(path):
    try:
        f = gh_api.get_file(OWNER, REPO, path, ref=BRANCH)
        if isinstance(f, dict) and f.get("content"):
            return base64.b64decode(f["content"]).decode("utf-8", errors="replace")
        return None
    except Exception as e:
        return f"ERR: {e}"

for p in sorted(paths):
    if p.startswith("synced/h24/bigeval_live/") and p.endswith(".jsonl"):
        raw = get_raw(p)
        if raw is None:
            print("LIVE", p, "MISSING/EMPTY")
        elif raw.startswith("ERR"):
            print("LIVE", p, raw)
        else:
            lines = [l for l in raw.splitlines() if l.strip()]
            print(f"LIVE {p} lines={len(lines)}")
            if lines:
                print("  first:", lines[0][:300])
                print("  last:", lines[-1][:300])

# 4. cal.out / launcher.log / chain log contents (full text, truncated at 4000 chars)
for p in sorted(paths):
    if p.endswith(("_cal.out", ".launcher.log", "_chain.log", "chain.log")):
        raw = get_raw(p)
        if raw is None:
            print("FILE", p, "MISSING/EMPTY")
        elif raw.startswith("ERR"):
            print("FILE", p, raw)
        else:
            print(f"FILE {p} bytes={len(raw)}")
            print(raw[:4000])
