"""H23 status 13:42 run: branch-head-only check (head unchanged => tree unchanged,
so cal.out sizes follow from the 12:14 full-tree read; quota rule).
Reads sync/h23 + sync/h29 heads and rate limit. 3 API calls.
Writes h23-status-1342.log next to this script.
"""
import json, sys, os
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

OWNER, REPO = "robertyang87", "carbon-league"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "h23-status-1342.log")

lines = ["MARKER H23STATUS1342"]
def emit(s):
    lines.append(s)
    print(s, flush=True)

for br_name in ["sync/h23", "sync/h29"]:
    try:
        br = req("GET", f"/repos/{OWNER}/{REPO}/branches/{br_name}")
        c = br["commit"]
        emit(f"BRANCH_HEAD {br_name} {c['sha'][:7]} {c['commit']['committer']['date']} {c['commit']['message'][:120]}")
    except Exception as e:
        emit(f"BRANCH_HEAD {br_name} ERROR {type(e).__name__}: {e}")

try:
    rl = req("GET", "/rate_limit")
    core = rl["resources"]["core"]
    emit(f"RATE_LIMIT core remaining={core['remaining']} reset={core['reset']}")
except Exception as e:
    emit(f"RATE_LIMIT ERROR {type(e).__name__}: {e}")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("WROTE", OUT)
