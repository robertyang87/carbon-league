#!/usr/bin/env python3
"""War-room-sync ~09:00 trip: verify gap2 corpus warehousing by API read-back.
Calls: main commits x1, 3 corpus dir listings, sync/h29 + sync/h31 heads, rate_limit x1."""
import sys, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {}
try:
    cs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha=main&per_page=5")
    out["main"] = [{"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"],
                    "msg": c["commit"]["message"].split("\n")[0][:90]} for c in cs]
except Exception as e:
    out["main_error"] = repr(e)

blobs = json.load(open("/home/hatch/workspace/carbon-live/corpus_staging_gap2/blob_shas.json"))
for label, dirname, expect in [
        ("hakimi3pro", "data/hakimi3pro_corpus", 161),
        ("fable6", "data/fable6_corpus", 203),
        ("k4v2", "data/k4v2_corpus", 306)]:
    try:
        items = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/{dirname}?ref=main")
        files = [i for i in items if i["type"] == "file"]
        byname = {i["name"]: i["size"] for i in files}
        # spot-check 2 new files from this family's blob_shas keys
        newnames = sorted(n.split("/", 1)[1] for n in blobs if n.startswith(label + "/"))[:2]
        checks = {n: byname.get(n) for n in newnames}
        out[label] = {"count": len(files), "expected": expect,
                      "match": len(files) == expect, "spot_sizes": checks}
    except Exception as e:
        out[label + "_error"] = repr(e)

for b in ["sync/h29", "sync/h31"]:
    try:
        cs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={b}&per_page=1")
        c = cs[0]
        out[b] = {"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"]}
    except Exception as e:
        out[b] = {"error": repr(e)}
try:
    rl = gh_api.req("GET", "/rate_limit")
    out["rate_core_remaining"] = rl["resources"]["core"]["remaining"]
    out["rate_core_reset"] = rl["resources"]["core"]["reset"]
except Exception as e:
    out["rate_error"] = repr(e)
print(json.dumps(out, indent=1, ensure_ascii=False))
