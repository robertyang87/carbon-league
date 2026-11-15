"""H23 status direct read driver. Marker: H23STATUS2127.
Reads sync/h23 branch head + latest commits + sizes of the probe/m7 files.
Usage: python3 h23-status-2127-driver.py
Writes h23-status-2127.log next to this script.
"""
import json, sys, os
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

MARKER = "H23STATUS2127"
OWNER, REPO = "robertyang87", "carbon-league"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h23-status-2127.log")

lines = [f"MARKER {MARKER}"]
def emit(s):
    lines.append(s)
    print(s, flush=True)

try:
    br = req("GET", f"/repos/{OWNER}/{REPO}/branches/sync/h23")
    c = br["commit"]
    emit(f"BRANCH_HEAD {c['sha'][:7]} {c['commit']['committer']['date']} {c['commit']['message'][:80]}")
except Exception as e:
    emit(f"BRANCH_HEAD ERROR {type(e).__name__}: {e}")

try:
    commits = req("GET", f"/repos/{OWNER}/{REPO}/commits?sha=sync/h23&per_page=3")
    for cm in commits:
        emit(f"COMMIT {cm['sha'][:7]} {cm['commit']['committer']['date']} {cm['commit']['message'][:80]}")
except Exception as e:
    emit(f"COMMITS ERROR {type(e).__name__}: {e}")

for path in [
    "synced/h23/bigeval_live/k4v2r1_e0.jsonl",
    "synced/h23/bigeval_live/k4v2g1_e0.jsonl",
    "synced/h23/k4probe.out",
    "synced/h23/k4v2r1_e0.launcher.log",
    "synced/h23/m7bsr1_cal.out",
    "synced/h23/bigeval_live/m7bsr1_m2.jsonl",
    "synced/h23/bigeval_live/m7bsr1_echo2.jsonl",
    "synced/h23/lineA_cal.out",
]:
    try:
        info = req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref=sync/h23")
        emit(f"FILE {path} size={info.get('size')} sha={info.get('sha','')[:7]}")
        if info.get("size", 0) and info.get("size", 0) <= 2000 and info.get("encoding") == "base64":
            import base64
            emit("CONTENT " + base64.b64decode(info["content"]).decode("utf-8", errors="replace")[:1500])
    except Exception as e:
        emit(f"FILE {path} ERROR {type(e).__name__}: {e}")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("WROTE", OUT)
