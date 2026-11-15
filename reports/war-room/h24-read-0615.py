import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
O, R = "robertyang87", "carbon-league"
br = gh_api.req("GET", f"/repos/{O}/{R}/branches/sync/h24")
sha = br["commit"]["sha"]
date = br["commit"]["commit"]["committer"]["date"]
print("BRANCH_HEAD", sha, date)
tree = gh_api.req("GET", f"/repos/{O}/{R}/git/trees/{sha}?recursive=1")
paths = [t["path"] for t in tree.get("tree", [])]
h24 = [p for p in paths if p.startswith("synced/h24/")]
live = [p for p in h24 if p.startswith("synced/h24/bigeval_live/")]
cal = [p for p in h24 if p.endswith("_cal.out")]
print("SYNCED_H24_COUNT", len(h24))
print("BIGEVAL_LIVE_COUNT", len(live))
print("CAL_OUT", cal)
for p in cal:
    f = gh_api.get_file(O, R, p, ref="sync/h24")
    print("CAL_BLOB", p, f.get("sha"), f.get("size"))
    print(base64.b64decode(f["content"]).decode("utf-8", errors="replace"))
# main head for war-room cross-check
mb = gh_api.req("GET", f"/repos/{O}/{R}/branches/main")
print("MAIN_HEAD", mb["commit"]["sha"], mb["commit"]["commit"]["committer"]["date"])
