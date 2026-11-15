#!/usr/bin/env python3
"""H24 status probe h24-0829: sync/h24 head, dir counts, latest live file, cal.out."""
import sys, json
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

# latest live file (wt_BrU6S.jsonl per prior trips) - full read
# NOTE: gh_api.get_file returns the contents-API dict (base64 content), not bytes.
import base64
def _decode(raw):
    if isinstance(raw, dict):
        return base64.b64decode(raw["content"]).decode("utf-8", "replace")
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw

try:
    raw = gh_api.get_file(OWNER, REPO, "synced/h24/bigeval_live/wt_BrU6S.jsonl",
                          ref="sync/h24")
    text = _decode(raw)
    lines = [l for l in text.splitlines() if l.strip()]
    import hashlib
    out["live_wt_BrU6S"] = {"lines": len(lines),
                            "blob_sha": raw.get("sha", "")[:8] if isinstance(raw, dict) else None,
                            "sha256": hashlib.sha256(text.encode()).hexdigest()[:8],
                            "last_line": lines[-1][:300] if lines else None}
except Exception as e:
    out["live_wt_BrU6S"] = {"error": str(e)[:300]}

# cal.out full text
try:
    raw = gh_api.get_file(OWNER, REPO, "synced/h24/m3s2r1_cal.out", ref="sync/h24")
    text = _decode(raw)
    import hashlib
    out["cal_out"] = {"blob_sha": raw.get("sha", "")[:8] if isinstance(raw, dict) else None,
                      "sha256": hashlib.sha256(text.encode()).hexdigest()[:8],
                      "text": text}
except Exception as e:
    out["cal_out"] = {"error": str(e)[:300]}

try:
    rl = gh_api.req("GET", "/rate_limit")
    core = rl["resources"]["core"]
    out["rate_limit"] = {"remaining": core["remaining"], "limit": core["limit"],
                         "reset": core["reset"]}
except Exception as e:
    out["rate_limit"] = {"error": str(e)[:200]}

print(json.dumps(out, indent=1, ensure_ascii=False))
