import base64
import hashlib
import json
import os
import sys

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER = "robertyang87"
REPO = "carbon-league"
OUTDIR = "/home/hatch/workspace/carbon-live/a4r1"
os.makedirs(OUTDIR, exist_ok=True)
FILES = {
    "m3s2r1_a4.py": "4a85855ba8878ec79d8c820f87e418cb",
    "m3s2r1_base.py": "11d8005930b298290c2e150949349647",
}
results = {}
for name, want_md5 in FILES.items():
    path = f"code/exp/{name}"
    meta = gh_api.get_file(OWNER, REPO, path, ref="main")
    blob = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/git/blobs/{meta['sha']}")
    raw = base64.b64decode(blob["content"])
    md5 = hashlib.md5(raw).hexdigest()
    if md5 != want_md5:
        raise SystemExit(f"MD5 MISMATCH {name}: got {md5}, want {want_md5}")
    out = os.path.join(OUTDIR, name)
    with open(out, "wb") as f:
        f.write(raw)
    results[name] = {"path": path, "blob_sha": meta["sha"], "size": len(raw), "md5": md5, "saved": out}
    print(json.dumps(results[name], indent=2))
with open(os.path.join(OUTDIR, "fetch.json"), "w") as f:
    json.dump(results, f, indent=2)
