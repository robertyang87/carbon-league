"""H23 status 15:57 run 2026-11-14: branch-head check for sync/h23 + sync/h29 + sync/h31 + main.
Heads unchanged => trees unchanged, follow from last full-tree read (15:27/15:42).
If a sync head moved, fetch tree to list new synced/<tag>/ entries.
Reads rate limit too. Writes h23-status-1557-1114.log next to this script.
NOTE: h23-status-1557.py/.log on disk are stale from 2026-11-13; do not reuse.
"""
import json, sys, os
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

OWNER, REPO = "robertyang87", "carbon-league"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "h23-status-1557-1114.log")
PREV = {"sync/h23": "bd950f2", "sync/h29": "16d2b91", "sync/h31": "bf965a1", "main": "907c1059"}

lines = ["MARKER H23STATUS1557-1114"]
def emit(s):
    lines.append(s)
    print(s, flush=True)

heads = {}
for br_name in ["sync/h23", "sync/h29", "sync/h31", "main"]:
    try:
        br = req("GET", f"/repos/{OWNER}/{REPO}/branches/{br_name}")
        c = br["commit"]
        heads[br_name] = c["sha"][:7]
        emit(f"BRANCH_HEAD {br_name} {c['sha'][:7]} {c['commit']['committer']['date']} {c['commit']['message'][:140]}")
    except Exception as e:
        emit(f"BRANCH_HEAD {br_name} ERROR {type(e).__name__}: {e}")

for br_name, prev in PREV.items():
    cur = heads.get(br_name)
    if not cur:
        continue
    if cur != prev:
        emit(f"{br_name} HEAD_CHANGED {prev} -> {cur}, fetching tree")
        try:
            tree = req("GET", f"/repos/{OWNER}/{REPO}/git/trees/{cur}?recursive=1")
            tag = br_name.split("/")[1] if "/" in br_name else None
            for x in tree.get("tree", []):
                p = x["path"]
                if tag and p.startswith(f"synced/{tag}/"):
                    emit(f"TREE {p} size={x.get('size')} sha={x['sha'][:7]}")
        except Exception as e:
            emit(f"TREE {br_name} ERROR {type(e).__name__}: {e}")
    else:
        emit(f"{br_name} HEAD_UNCHANGED since {prev}")

if heads.get("sync/h23") and heads.get("sync/h23") != PREV["sync/h23"]:
    for name in ["lineA_cal.out", "m3s2r1pre_cal.out", "m7bsr1_cal.out"]:
        try:
            blob = req("GET", f"/repos/{OWNER}/{REPO}/contents/synced/h23/{name}?ref=sync/h23")
            import base64
            content = base64.b64decode(blob["content"]).decode("utf-8", "replace")
            emit(f"CAL {name}: {content[:400]!r}")
        except Exception as e:
            emit(f"CAL {name} ERROR {type(e).__name__}: {e}")

try:
    rl = req("GET", "/rate_limit")
    core = rl["resources"]["core"]
    emit(f"RATE_LIMIT core remaining={core['remaining']} reset={core['reset']}")
except Exception as e:
    emit(f"RATE_LIMIT ERROR {type(e).__name__}: {e}")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("WROTE", OUT)
