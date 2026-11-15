#!/usr/bin/env python3
"""H24 status probe h24-1344: sync/h24 branch head, dir listings,
m3s2r1_cal.out full text, m3s2r1_echo2.jsonl row count / wins / mean / last game."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
REF = "sync/h24"

def branch_head(branch):
    r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{branch}")
    c = r["commit"]["commit"]
    return {"sha": r["commit"]["sha"], "date": c["committer"]["date"],
            "msg": c["message"].splitlines()[0][:120]}

def list_dir(path, ref):
    r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={ref}")
    if isinstance(r, dict):
        return r
    return [{"name": e["name"], "size": e.get("size", 0), "sha": e.get("sha", "")[:12],
             "type": e["type"]} for e in r]

def get_text(path, ref):
    r = gh_api.get_file(OWNER, REPO, path, ref=ref)
    if isinstance(r, dict) and "content" in r:
        return base64.b64decode(r["content"]).decode("utf-8", errors="replace"), r.get("sha", "")[:12]
    return None, (r if isinstance(r, dict) else str(r))[:200]

out = {"probe": "h24-1344"}
out["branch_head"] = branch_head(REF)
top = list_dir("synced/h24", REF)
out["synced_h24"] = top if isinstance(top, dict) else {
    "count": len(top),
    "cal_outs": [i for i in top if i["name"].endswith("_cal.out")]}
live = list_dir("synced/h24/bigeval_live", REF)
out["bigeval_live"] = live if isinstance(live, dict) else {
    "count": len(live),
    "echo2_entry": [i for i in live if i["name"] == "m3s2r1_echo2.jsonl"]}
cal, cal_sha = get_text("synced/h24/m3s2r1_cal.out", REF)
out["cal_out"] = {"sha": cal_sha, "text": cal}
echo, echo_sha = get_text("synced/h24/bigeval_live/m3s2r1_echo2.jsonl", REF)
if echo is not None:
    lines = [l for l in echo.splitlines() if l.strip()]
    wins = 0; margins = []; last = None
    for l in lines:
        try:
            d = json.loads(l)
        except Exception:
            continue
        m = d.get("margin")
        if isinstance(m, (int, float)):
            margins.append(m)
            if m > 0:
                wins += 1
        last = d
    out["echo2"] = {"sha": echo_sha, "n": len(lines), "wins": wins,
                    "mean": round(sum(margins) / len(margins), 1) if margins else None,
                    "last_seed": last.get("seed") if last else None,
                    "last_time": (last.get("ts") or last.get("time") or last.get("finished_at")) if last else None,
                    "last_margin": last.get("margin") if last else None,
                    "last_keys": sorted(last.keys()) if last else None}
else:
    out["echo2"] = {"error": echo_sha}
try:
    rl = gh_api.req("GET", "/rate_limit")
    out["rate_limit_core_remaining"] = rl["resources"]["core"]["remaining"]
except Exception as e:
    out["rate_limit_core_remaining"] = f"error {e}"[:120]

print(json.dumps(out, indent=1, ensure_ascii=False))
