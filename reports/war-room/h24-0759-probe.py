#!/usr/bin/env python3
"""H24 status probe h24-0759: sync/h24 head, dir counts, latest live file, cal.out."""
import sys, json, base64, hashlib
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {}

try:
    r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/sync/h24")
    c = r["commit"]["commit"]
    out["head"] = {"sha": r["commit"]["sha"][:8],
                   "date": c["committer"]["date"],
                   "msg": c["message"].splitlines()[0][:100]}
except Exception as e:
    out["head"] = {"error": str(e)[:200]}

def list_dir(path, ref):
    r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={ref}")
    if isinstance(r, dict):
        return r
    return [{"name": e["name"], "size": e.get("size", 0)} for e in r]

for path in ["synced/h24", "synced/h24/bigeval_live"]:
    try:
        items = list_dir(path, "sync/h24")
        if isinstance(items, dict):
            out[path] = {"error": items}
        else:
            cal = [i["name"] for i in items if i["name"].endswith("_cal.out")]
            out[path] = {"count": len(items), "cal_outs": cal,
                         "names_tail": [i["name"] for i in items[-5:]]}
    except Exception as e:
        out[path] = {"error": str(e)[:200]}

for label, path in [("live_wt_BrU6S", "synced/h24/bigeval_live/wt_BrU6S.jsonl"),
                    ("cal_out", "synced/h24/m3s2r1_cal.out")]:
    try:
        d = gh_api.get_file(OWNER, REPO, path, ref="sync/h24")
        text = base64.b64decode(d["content"]).decode("utf-8", "replace")
        lines = [l for l in text.splitlines() if l.strip()]
        out[label] = {"blob_sha": d.get("sha", "")[:8],
                      "size": d.get("size"),
                      "lines": len(lines),
                      "sha256": hashlib.sha256(text.encode()).hexdigest()[:8],
                      "last_line": lines[-1][:300] if lines else None}
        if label == "cal_out":
            out[label]["text"] = text
    except Exception as e:
        out[label] = {"error": str(e)[:300]}

try:
    rl = gh_api.req("GET", "/rate_limit")
    core = rl["resources"]["core"]
    out["rate_limit"] = {"remaining": core["remaining"], "limit": core["limit"],
                         "reset": core["reset"]}
except Exception as e:
    out["rate_limit"] = {"error": str(e)[:200]}

print(json.dumps(out, indent=1, ensure_ascii=False))
