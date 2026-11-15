import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

owner, repo = "robertyang87", "carbon-league"
branch = "sync/h24"

r = gh_api.req("GET", f"/repos/{owner}/{repo}/commits?sha={branch}&per_page=5")
print("== latest sync/h24 commits (UTC) ==")
for c in r:
    print(c["commit"]["committer"]["date"], c["sha"][:8])

r = gh_api.req("GET", f"/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")
paths = [t["path"] for t in r.get("tree", [])]
print("== total paths:", len(paths))
print("== candidates (e0 / m3s2r1 / echo2 / cal) ==")
for p in sorted(paths):
    lp = p.lower()
    if "cal" in lp or "e0" in lp or "m3s2r1" in lp or "echo2" in lp:
        print(p)
