"""H24 status read 2026-11-14 ~07:30 CST: sync/h24 branch head + subtree counts + cal.out blobs + main head. Quota-light: tree walk via tree SHAs, no full re-fetch when blob SHA unchanged."""
import sys, json
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

OWNER, REPO = "robertyang87", "carbon-league"

b = req("GET", f"/repos/{OWNER}/{REPO}/branches/sync/h24")
head = b["commit"]["sha"]
date = b["commit"]["commit"]["committer"]["date"]
print("sync/h24 head:", head, "committed:", date)
print("sync/h24 msg:", b["commit"]["commit"]["message"][:100])

root_tree = req("GET", f"/repos/{OWNER}/{REPO}/git/trees/{b['commit']['commit']['tree']['sha']}")
synced_sha = next(e["sha"] for e in root_tree["tree"] if e["path"] == "synced")
synced_tree = req("GET", f"/repos/{OWNER}/{REPO}/git/trees/{synced_sha}")
h24_sha = next(e["sha"] for e in synced_tree["tree"] if e["path"] == "h24")
h24_tree = req("GET", f"/repos/{OWNER}/{REPO}/git/trees/{h24_sha}?recursive=1")
entries = h24_tree.get("tree", [])
print("synced/h24 recursive entries:", len(entries), "truncated:", h24_tree.get("truncated"))
live = [e for e in entries if e["path"].startswith("bigeval_live/")]
cals = [e for e in entries if e["path"].endswith("_cal.out")]
print("bigeval_live items:", len(live))
print("cal.out files:", [(e["path"], e["sha"], e.get("size")) for e in cals])
for e in cals:
    f = req("GET", f"/repos/{OWNER}/{REPO}/contents/synced/h24/{e['path']}?ref=sync/h24")
    import base64
    print("---", e["path"], "---")
    print(base64.b64decode(f["content"]).decode(errors="replace"))

m = req("GET", f"/repos/{OWNER}/{REPO}/branches/main")
print("main head:", m["commit"]["sha"], "committed:", m["commit"]["commit"]["committer"]["date"])
