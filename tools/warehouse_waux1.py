#!/usr/bin/env python3
"""Warehouse m9xj570waux1.py pack to main via Git Data API (blob->tree->commit->ref),
then read back and verify md5."""
import sys, base64, hashlib
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req

OWNER, REPO = "robertyang87", "carbon-league"
PATH = "code/exp/m9xj570waux1.py"
LOCAL = "/home/hatch/workspace/carbon-incoming/m9xj570waux1.py"
EXPECT_MD5 = "9962e84aded27aa554b153a92fe8e9f5"

data = open(LOCAL, "rb").read()
assert hashlib.md5(data).hexdigest() == EXPECT_MD5, "local md5 mismatch"

blob = req("POST", f"/repos/{OWNER}/{REPO}/git/blobs",
           {"content": base64.b64encode(data).decode(), "encoding": "base64"})
print("blob:", blob["sha"])

head = req("GET", f"/repos/{OWNER}/{REPO}/git/refs/heads/main")
parent = head["object"]["sha"]
print("parent:", parent)
commit0 = req("GET", f"/repos/{OWNER}/{REPO}/git/commits/{parent}")
base_tree = commit0["tree"]["sha"]

tree = req("POST", f"/repos/{OWNER}/{REPO}/git/trees",
           {"base_tree": base_tree,
            "tree": [{"path": PATH, "mode": "100644", "type": "blob", "sha": blob["sha"]}]})
print("tree:", tree["sha"])

commit = req("POST", f"/repos/{OWNER}/{REPO}/git/commits",
             {"message": "pack: m9xj570waux1 (it570 aux-head arm, S15 cheap-improvement round 2/2)",
              "tree": tree["sha"], "parents": [parent]})
print("commit:", commit["sha"])

req("PATCH", f"/repos/{OWNER}/{REPO}/git/refs/heads/main", {"sha": commit["sha"]})
print("ref updated")

# read-back: fetch blob by sha and verify md5
rb = req("GET", f"/repos/{OWNER}/{REPO}/git/blobs/{blob['sha']}")
raw = base64.b64decode(rb["content"])
print("readback bytes:", len(raw), "md5:", hashlib.md5(raw).hexdigest())
assert len(raw) == len(data) and hashlib.md5(raw).hexdigest() == EXPECT_MD5
print("WAREHOUSE OK")
