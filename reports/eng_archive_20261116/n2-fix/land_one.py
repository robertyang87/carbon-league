#!/usr/bin/env python3
"""Land ONE file to carbon-league main with pre/post md5 verification.

Usage: land_one.py <repo-path> <local-file> <expect-base-md5> <expect-new-md5> <message>
Aborts before PUT if the current main bytes are not the expected base.
"""
import base64
import hashlib
import json
import sys

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
path, local, base_md5, new_md5, msg = sys.argv[1:6]

cur = gh_api.get_file(OWNER, REPO, path, ref="main")
cur_bytes = base64.b64decode(cur["content"])
cur_md5 = hashlib.md5(cur_bytes).hexdigest()
print(json.dumps({"step": "pre", "path": path, "sha": cur["sha"], "md5": cur_md5}))
if cur_md5 != base_md5:
    print(json.dumps({"ABORT": f"main md5 {cur_md5} != expected base {base_md5}"}))
    sys.exit(2)

new_bytes = open(local, "rb").read()
assert hashlib.md5(new_bytes).hexdigest() == new_md5, "local file md5 mismatch"
res = gh_api.put_file(OWNER, REPO, path, new_bytes, msg, sha=cur["sha"], branch="main")
print(json.dumps({"step": "put", "commit": (res.get("commit") or {}).get("sha"),
                  "content_sha": (res.get("content") or {}).get("sha")}))

back = gh_api.get_file(OWNER, REPO, path, ref="main")
back_bytes = base64.b64decode(back["content"])
back_md5 = hashlib.md5(back_bytes).hexdigest()
print(json.dumps({"step": "post", "path": path, "sha": back["sha"], "md5": back_md5,
                  "bytes": len(back_bytes), "match": back_md5 == new_md5}))
if back_md5 != new_md5:
    sys.exit(3)
