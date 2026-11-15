#!/usr/bin/env python3
"""Newface watch 23:56 run — GitHub read batch (single trip):
list data/ dir, read matches files for changed labels, count corpus dirs,
list ladder-replay-pull recent runs. Prints compact JSON."""
import json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"

out = {}
# data/ listing
try:
    listing = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/data?ref=main")
    out["data_listing"] = [e["name"] for e in listing] if isinstance(listing, list) else {"error": str(listing)[:200]}
except Exception as e:
    out["data_listing"] = {"error": str(e)[:300]}

labels = ["krypton", "lite-v8", "hakimi3pro", "hakimi31", "m3s3z", "fable", "hakimi3", "m9xj570"]
out["matches"] = {}
for lab in labels:
    try:
        f = gh_api.get_file(OWNER, REPO, f"data/{lab}_matches.json", ref="main")
        if isinstance(f, dict) and "content" in f:
            import base64
            raw = base64.b64decode(f["content"])
            arr = json.loads(raw)
            out["matches"][lab] = {"count": len(arr), "sha": f["sha"][:8], "size": f["size"]}
        else:
            out["matches"][lab] = {"error": str(f)[:200]}
    except Exception as e:
        out["matches"][lab] = {"error": str(e)[:300]}

out["corpus"] = {}
for lab in ["m3s3z", "hakimi31", "hakimi3", "hakimi3pro", "lite-v8", "krypton", "fable", "k4v3", "fable6"]:
    try:
        lst = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/data/{lab}_corpus?ref=main")
        if isinstance(lst, list):
            out["corpus"][lab] = len([e for e in lst if e["type"] == "file"])
        else:
            out["corpus"][lab] = {"error": str(lst)[:150]}
    except Exception as e:
        out["corpus"][lab] = {"error": str(e)[:200]}

try:
    runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/ladder-replay-pull.yml/runs?per_page=5")
    out["workflow_runs"] = [
        {"id": r["id"], "status": r["status"], "conclusion": r["conclusion"],
         "created_at": r["created_at"], "display_title": r.get("display_title", "")[:80]}
        for r in runs.get("workflow_runs", [])]
except Exception as e:
    out["workflow_runs"] = {"error": str(e)[:300]}

print(json.dumps(out, ensure_ascii=False, indent=1))
