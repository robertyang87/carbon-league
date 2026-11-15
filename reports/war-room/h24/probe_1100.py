#!/usr/bin/env python3
"""H24 11:00 status probe: sync/h24 head + dir counts + m3s2r1_cal.out + Echo2 live archive line count."""
import sys, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {}

def branch_head(branch):
    r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{branch}")
    c = r["commit"]["commit"]
    return {"sha": r["commit"]["sha"][:8], "date": c["committer"]["date"],
            "msg": c["message"].splitlines()[0][:100]}

def list_dir(path, ref):
    r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={ref}")
    if isinstance(r, dict):
        return r
    return [{"name": e["name"], "size": e.get("size", 0), "type": e["type"]} for e in r]

try:
    out["sync_h24_head"] = branch_head("sync/h24")
except Exception as e:
    out["sync_h24_head"] = {"error": str(e)[:200]}

try:
    items = list_dir("synced/h24", "sync/h24")
    if isinstance(items, dict):
        out["synced_h24"] = {"error": items}
    else:
        cal = [i for i in items if i["name"].endswith("_cal.out")]
        out["synced_h24"] = {"count": len(items), "cal_outs": [(i["name"], i["size"]) for i in cal]}
except Exception as e:
    out["synced_h24"] = {"error": str(e)[:200]}

try:
    items = list_dir("synced/h24/bigeval_live", "sync/h24")
    if isinstance(items, dict):
        out["bigeval_live"] = {"error": items}
    else:
        out["bigeval_live"] = {"count": len(items)}
except Exception as e:
    out["bigeval_live"] = {"error": str(e)[:200]}

# cal.out full text (expect unchanged: 683 B, CAL_ALL_DONE)
try:
    f = gh_api.get_file(OWNER, REPO, "synced/h24/m3s2r1_cal.out", ref="sync/h24")
    txt = f.decode() if isinstance(f, (bytes, bytearray)) else str(f)
    out["m3s2r1_cal_out"] = {"bytes": len(txt), "text": txt}
except Exception as e:
    out["m3s2r1_cal_out"] = {"error": str(e)[:300]}

# Echo2 live archive single-file check: line count + tail
try:
    f = gh_api.get_file(OWNER, REPO, "synced/h24/bigeval_live/m3s2r1_echo2.jsonl", ref="sync/h24")
    txt = f.decode() if isinstance(f, (bytes, bytearray)) else str(f)
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    last = json.loads(lines[-1]) if lines else {}
    out["echo2_live"] = {"lines": len(lines), "last": last}
except Exception as e:
    out["echo2_live"] = {"error": str(e)[:300]}

print(json.dumps(out, indent=1, ensure_ascii=False))
