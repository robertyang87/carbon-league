import base64
import hashlib
import sys

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

EXPECT_BLOB = "79514aeb0c835f3038520ce52baae262f3f3935c"
EXPECT_SIZE = 3728418

r = gh_api.req("GET", f"/repos/robertyang87/carbon-league/git/blobs/{EXPECT_BLOB}")
print("keys:", sorted(r.keys()))
print("encoding:", r.get("encoding"), "size:", r.get("size"), "sha:", r.get("sha"))
raw = base64.b64decode("".join(r["content"].split()))
print("bytes:", len(raw))
print("md5:", hashlib.md5(raw).hexdigest())
blob = hashlib.sha1(b"blob %d\x00" % len(raw) + raw).hexdigest()
print("blob sha:", blob)
with open("/home/hatch/workspace/carbon-live/probe/m3s2g1_base.py", "wb") as f:
    f.write(raw)
print("EXPECT blob", EXPECT_BLOB, "size", EXPECT_SIZE)
print("MATCH:", blob == EXPECT_BLOB and len(raw) == EXPECT_SIZE)
