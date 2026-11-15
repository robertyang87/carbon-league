"""H23 status 20:12 run (2026-11-14): branch-head check for sync/h23 + sync/h29 + sync/h31 + main.
Prev heads (19:27/19:28 trip + war-room 19:33-19:41 sync):
  h23 bd950f26 (full bd950f26eec15f3fd4c457aafaeb69ef5f18d8cd @ 2026-11-12 20:42:01 CST)
  h29 77f7d26, h31 a7e718c9, main 85e508b2 (@19:30 CST, newface-watch).
Compare 7-char prefixes only (2026-11-14 18:57 prefix-mix lesson).
If a sync head moved, fetch its tree. Verify the three h23 cal.out files by direct read.
Writes h23-status-2012-1114.log. Marker: H23STAT2012.
"""
import sys, os, base64
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

OWNER, REPO = "robertyang87", "carbon-league"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "h23-status-2012-1114.log")
PREV = {"sync/h23": "bd950f2", "sync/h29": "77f7d26", "sync/h31": "a7e718c", "main": "97ece4f"}

lines = ["MARKER H23STAT2012"]
def emit(s):
    lines.append(s)
    print(s, flush=True)

heads = {}
full = {}
for br_name in ["sync/h23", "sync/h29", "sync/h31", "main"]:
    try:
        br = req("GET", f"/repos/{OWNER}/{REPO}/branches/{br_name}")
        c = br["commit"]
        heads[br_name] = c["sha"][:7]
        full[br_name] = c["sha"]
        emit(f"BRANCH_HEAD {br_name} {c['sha']} {c['commit']['committer']['date']} {c['commit']['message'][:140]}")
    except Exception as e:
        emit(f"BRANCH_HEAD {br_name} ERROR {type(e).__name__}: {e}")

for br_name, prev in PREV.items():
    cur = heads.get(br_name)
    if not cur:
        continue
    if cur != prev:
        emit(f"{br_name} HEAD_CHANGED {prev} -> {cur}")
        if br_name != "main":
            try:
                tree = req("GET", f"/repos/{OWNER}/{REPO}/git/trees/{full[br_name]}?recursive=1")
                tag = br_name.split("/")[1]
                for x in tree.get("tree", []):
                    p = x["path"]
                    if p.startswith(f"synced/{tag}/"):
                        emit(f"TREE {p} size={x.get('size')} sha={x['sha'][:7]}")
            except Exception as e:
                emit(f"TREE {br_name} ERROR {type(e).__name__}: {e}")
    else:
        emit(f"{br_name} HEAD_UNCHANGED since {prev}")

# bigeval_live count via one tree fetch (single call; do NOT per-file traverse)
try:
    tree = req("GET", f"/repos/{OWNER}/{REPO}/git/trees/{full.get('sync/h23')}?recursive=1")
    live = [x for x in tree.get("tree", []) if x["path"].startswith("synced/h23/bigeval_live/")]
    emit(f"LIVE synced/h23/bigeval_live entries={len(live)}")
except Exception as e:
    emit(f"LIVE ERROR {type(e).__name__}: {e}")

# Verify the three cal.out files' content via direct read (only if h23 head unchanged).
if heads.get("sync/h23") == PREV["sync/h23"]:
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
