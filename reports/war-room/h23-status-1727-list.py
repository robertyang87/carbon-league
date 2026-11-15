"""h23-status 17:27 probe: read sync/h23 branch state (step 1: list)."""
import sys, json
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req, get_file

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h23"

commits = req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=3")
for c in commits:
    print("COMMIT", c["sha"][:7], c["commit"]["committer"]["date"], c["commit"]["message"].splitlines()[0][:100])

live_dir = get_file(OWNER, REPO, "synced/h23/bigeval_live", ref=BRANCH)
print("LIVE_DIR:")
for e in live_dir if isinstance(live_dir, list) else []:
    print(" ", e["name"], e["size"])

top = get_file(OWNER, REPO, "synced/h23", ref=BRANCH)
print("TOP_DIR:")
for e in top if isinstance(top, list) else []:
    print(" ", e["name"], e["size"], e["type"])
