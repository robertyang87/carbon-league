import json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
OWNER, REPO = "robertyang87", "carbon-league"
commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha=sync/h29&per_page=8")
for c in commits:
    print(c["sha"][:8], c["commit"]["author"]["date"], c["commit"]["message"][:80])
print("=== e716ee33 files ===")
c = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits/e716ee3397f03d17eb804243ea4516851a48861f")
for f in c.get("files", []):
    print(f["filename"], f.get("status"), "add", f.get("additions"), "del", f.get("deletions"))
print("=== 4d636da2 ===")
try:
    c2 = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits/4d636da2")
    print(c2["sha"][:8], c2["commit"]["author"]["date"], c2["commit"]["message"][:80])
    for f in c2.get("files", []):
        print(f["filename"], f.get("status"), "add", f.get("additions"), "del", f.get("deletions"))
except Exception as e:
    print("4d636da2 lookup failed:", e)
