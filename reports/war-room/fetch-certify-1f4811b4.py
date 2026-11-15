import base64
import hashlib
import json
import os
import sys

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER = "robertyang87"
REPO = "carbon-league"
REF = "5b2b850f"
PATH = "code/certify_pack.py"
OUTDIR = "/home/hatch/workspace/carbon-live/update-certify-1f4811b4/code"
os.makedirs(OUTDIR, exist_ok=True)
meta = gh_api.get_file(OWNER, REPO, PATH, ref=REF)
raw = base64.b64decode(meta["content"])
out = os.path.join(OUTDIR, "certify_pack.py")
with open(out, "wb") as f:
    f.write(raw)
result = {
    "path": PATH,
    "ref": REF,
    "api_sha": meta.get("sha"),
    "size": len(raw),
    "md5": hashlib.md5(raw).hexdigest(),
    "saved": out,
}
with open("/home/hatch/workspace/carbon-live/update-certify-1f4811b4/fetch.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
