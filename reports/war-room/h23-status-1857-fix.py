"""H23 18:57 correction: full-SHA head compare + cal.out re-read (fixes 7-vs-8 char prefix bug in h23-status-1857-1114.py)."""
import sys, os, base64
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req
OWNER, REPO = "robertyang87", "carbon-league"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "h23-status-1857-1114.log")
PREV8 = {"sync/h23": "bd950f26", "sync/h29": "77f7d26", "sync/h31": "bf0ffd2", "main": "558a14e4"}
lines = []
with open(OUT) as f:
    lines = f.read().rstrip("\n").split("\n")
def emit(s):
    lines.append(s); print(s, flush=True)
emit("CORRECTION H23STAT1857-FIX: prior script compared 7-char slice vs 8-char prev -> false HEAD_CHANGED; re-check with full SHA")
for br_name, prev8 in PREV8.items():
    try:
        br = req("GET", f"/repos/{OWNER}/{REPO}/branches/{br_name}")
        sha = br["commit"]["sha"]
        emit(f"FULLSHA {br_name} {sha} prefix8={sha[:8]} prev8={prev8} match={sha[:8]==prev8}")
    except Exception as e:
        emit(f"FULLSHA {br_name} ERROR {type(e).__name__}: {e}")
for fn in ["lineA_cal.out", "m3s2r1pre_cal.out", "m7bsr1_cal.out"]:
    try:
        content = req("GET", f"/repos/{OWNER}/{REPO}/contents/synced/h23/{fn}?ref=sync/h23")
        raw = base64.b64decode(content["content"]).decode("utf-8", "replace")
        emit(f"CALOUT {fn} sha={content['sha'][:7]} len={len(raw)} first200={raw[:200]!r}")
    except Exception as e:
        emit(f"CALOUT {fn} ERROR {type(e).__name__}: {e}")
try:
    rl = req("GET", "/rate_limit")
    core = rl["resources"]["core"]
    emit(f"RATE_LIMIT core remaining={core['remaining']} reset={core['reset']}")
except Exception as e:
    emit(f"RATE_LIMIT ERROR {type(e).__name__}: {e}")
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("WROTE", OUT)
