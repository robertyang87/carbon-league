"""H23 status direct read driver. Marker: H23STATUS1312.
Reads sync/h23 + sync/h29 branch heads, sync/h23 tree (sizes, one call),
and rate limit. Usage: python3 h23-status-1312-driver.py
Writes h23-status-1312.log next to this script.
"""
import json, sys, os
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

MARKER = "H23STATUS1312"
OWNER, REPO = "robertyang87", "carbon-league"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "h23-status-1312.log")

lines = [f"MARKER {MARKER}"]
def emit(s):
    lines.append(s)
    print(s, flush=True)

for br_name in ["sync/h23", "sync/h29", "sync/h31", "main"]:
    try:
        br = req("GET", f"/repos/{OWNER}/{REPO}/branches/{br_name}")
        c = br["commit"]
        emit(f"BRANCH_HEAD {br_name} {c['sha'][:7]} {c['commit']['committer']['date']} {c['commit']['message'][:80]}")
    except Exception as e:
        emit(f"BRANCH_HEAD {br_name} ERROR {type(e).__name__}: {e}")

try:
    commits = req("GET", f"/repos/{OWNER}/{REPO}/commits?sha=sync/h23&per_page=3")
    for cm in commits:
        emit(f"COMMIT h23 {cm['sha'][:7]} {cm['commit']['committer']['date']} {cm['commit']['message'][:80]}")
except Exception as e:
    emit(f"COMMITS h23 ERROR {type(e).__name__}: {e}")

try:
    tree = req("GET", f"/repos/{OWNER}/{REPO}/git/trees/sync/h23?recursive=1")
    for item in tree.get("tree", []):
        if item["type"] == "blob":
            emit(f"TREE {item['path']} size={item.get('size')} sha={item.get('sha','')[:7]}")
except Exception as e:
    emit(f"TREE h23 ERROR {type(e).__name__}: {e}")

# Small cal.out files: fetch content only if non-empty per tree (sizes known above).
# Re-read tree sizes from lines is fragile; just attempt the known small ones.
for path in ["synced/h23/m7bsr1_cal.out", "synced/h23/lineA_cal.out", "synced/h23/m3s2r1pre_cal.out"]:
    try:
        info = req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref=sync/h23")
        size = info.get("size", 0)
        emit(f"FILE {path} size={size} sha={info.get('sha','')[:7]}")
        if size and size <= 2000 and info.get("encoding") == "base64":
            import base64
            emit("CONTENT " + base64.b64decode(info["content"]).decode("utf-8", errors="replace")[:1500])
    except Exception as e:
        emit(f"FILE {path} ERROR {type(e).__name__}: {e}")

try:
    rl = req("GET", "/rate_limit")
    core = rl["resources"]["core"]
    emit(f"RATE_LIMIT core remaining={core['remaining']} reset={core['reset']}")
except Exception as e:
    emit(f"RATE_LIMIT ERROR {type(e).__name__}: {e}")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("WROTE", OUT)
