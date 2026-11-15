"""H23 18:57 follow-up: inspect sync/h31 head move bf0ffd2 -> a7e718c9 (real change, not prefix bug)."""
import sys, os
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req
OWNER, REPO = "robertyang87", "carbon-league"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "h23-status-1857-1114.log")
def emit(s):
    print(s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")
try:
    c = req("GET", f"/repos/{OWNER}/{REPO}/commits/a7e718c971c40817b35aef3dea35d7547954715f")
    emit(f"H31_NEW_COMMIT a7e718c9 date={c['commit']['committer']['date']} msg={c['commit']['message'][:200]!r}")
    for f in c.get("files", [])[:30]:
        emit(f"H31_FILE {f['filename']} status={f['status']} additions={f['additions']} deletions={f['deletions']}")
except Exception as e:
    emit(f"H31_COMMIT ERROR {type(e).__name__}: {e}")
try:
    old = req("GET", f"/repos/{OWNER}/{REPO}/commits/bf0ffd2")
    emit(f"H31_OLD_COMMIT bf0ffd2 date={old['commit']['committer']['date']} msg={old['commit']['message'][:200]!r}")
except Exception as e:
    emit(f"H31_OLD ERROR {type(e).__name__}: {e}")
