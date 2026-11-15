#!/usr/bin/env python3
"""h23 status check 06:42 trip: sync/h23 + main heads, synced/h23 dir, cal.out contents,
sync/h29 head, main corpus warehousing counts (k4v3/fable6/hakimi3pro), recent main commits."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"

def head_commit(branch):
    cs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={branch}&per_page=1")
    c = cs[0]
    return {"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"],
            "msg": c["commit"]["message"][:100]}

def get_text(path, ref):
    d = gh_api.get_file(OWNER, REPO, path, ref=ref)
    if isinstance(d, dict) and d.get("encoding") == "base64":
        return base64.b64decode(d["content"]).decode("utf-8", errors="replace")
    return None

out = {}
try:
    rl = gh_api.req("GET", "/rate_limit")
    out["rate_remaining"] = rl["resources"]["core"]["remaining"]
    out["rate_reset"] = rl["resources"]["core"]["reset"]
except Exception as e:
    out["rate_error"] = str(e)

for br in ("main", "sync/h23", "sync/h29"):
    try:
        out[br] = head_commit(br)
    except Exception as e:
        out[br] = {"error": str(e)[:200]}

try:
    d = gh_api.get_file(OWNER, REPO, "synced/h23", ref="sync/h23")
    if isinstance(d, list):
        out["h23_dir_count"] = len(d)
        cals = [e["name"] for e in d if e["name"].endswith("_cal.out")]
        out["h23_cal_out"] = cals
        for c in cals:
            try:
                out[f"cal:{c}"] = get_text(f"synced/h23/{c}", "sync/h23")
            except Exception as e:
                out[f"cal:{c}"] = {"error": str(e)[:200]}
    else:
        out["h23_dir"] = {"error": str(d)[:200]}
except Exception as e:
    out["h23_dir"] = {"error": str(e)[:200]}

try:
    d = gh_api.get_file(OWNER, REPO, "synced/h23/bigeval_live", ref="sync/h23")
    out["h23_live_count"] = len(d) if isinstance(d, list) else {"error": str(d)[:200]}
except Exception as e:
    out["h23_live"] = {"error": str(e)[:200]}

# corpus warehousing on main
for label in ("k4v3", "fable6", "hakimi3pro"):
    try:
        d = gh_api.get_file(OWNER, REPO, f"data/{label}_corpus", ref="main")
        out[f"corpus:{label}"] = len(d) if isinstance(d, list) else {"error": str(d)[:150]}
    except Exception as e:
        out[f"corpus:{label}"] = {"error": str(e)[:150]}

# recent main commits since 04:28
try:
    cs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha=main&per_page=8")
    out["main_recent"] = [{"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"],
                           "msg": c["commit"]["message"][:90]} for c in cs]
except Exception as e:
    out["main_recent"] = {"error": str(e)[:200]}

print(json.dumps(out, indent=1, ensure_ascii=False))
