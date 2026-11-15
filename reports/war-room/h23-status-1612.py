"""H23 status 16:12 run: branch-head check for sync/h23 + sync/h29 + sync/h31.
Prev heads from 15:57 run (h23-status-last.json). If a head moved, fetch its
tree to list new synced/<tag>/ files. Writes h23-status-1612.log.
"""
import sys, os
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

OWNER, REPO = "robertyang87", "carbon-league"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "h23-status-1612.log")
PREV = {"sync/h23": "bd950f2", "sync/h29": "163ff0c", "sync/h31": "bf965a1"}

lines = ["MARKER H23STATUS1612"]
def emit(s):
    lines.append(s)
    print(s, flush=True)

heads = {}
for br_name in ["sync/h23", "sync/h29", "sync/h31"]:
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
            tag = br_name.split("/")[1]
            for x in tree.get("tree", []):
                p = x["path"]
                if p.startswith(f"synced/{tag}/"):
                    emit(f"TREE {p} size={x.get('size')} sha={x['sha'][:7]}")
        except Exception as e:
            emit(f"TREE {br_name} ERROR {type(e).__name__}: {e}")
    else:
        emit(f"{br_name} HEAD_UNCHANGED since {prev}")

try:
    rl = req("GET", "/rate_limit")
    core = rl["resources"]["core"]
    emit(f"RATE_LIMIT core remaining={core['remaining']} reset={core['reset']}")
except Exception as e:
    emit(f"RATE_LIMIT ERROR {type(e).__name__}: {e}")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("WROTE", OUT)
