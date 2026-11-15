"""h23 status check: read sync/h23 branch head + synced/h23 listing via GitHub API."""
import sys, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
BRANCH = "sync/h23"

# 1. branch head
br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{BRANCH}")
print("BRANCH_HEAD:", br["commit"]["sha"], br["commit"]["commit"]["author"]["date"],
      "|", br["commit"]["commit"]["message"].splitlines()[0][:120])

# 2. list synced/h23/ (root) - names + sizes only
root = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/synced/h23?ref={BRANCH}")
if isinstance(root, dict):
    print("ROOT:", root.get("message"))
else:
    print(f"ROOT entries: {len(root)}")
    for e in root:
        print("  ", e["type"], e["name"], e.get("size"))

# 3. list synced/h23/bigeval_live/ if present
try:
    live = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/synced/h23/bigeval_live?ref={BRANCH}")
    if isinstance(live, dict):
        print("LIVE_DIR:", live.get("message"))
    else:
        print(f"LIVE entries: {len(live)}")
        for e in live:
            print("  ", e["type"], e["name"], e.get("size"))
except Exception as ex:
    print("LIVE_DIR error:", ex)

# 4. rate limit remaining
rl = gh_api.req("GET", "/rate_limit")
print("RATE:", rl["resources"]["core"]["remaining"], "/", rl["resources"]["core"]["limit"])
