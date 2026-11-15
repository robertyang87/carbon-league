#!/usr/bin/env python3
"""h23 status check trip 2026-11-15 00:12 CST. Direct GitHub API reads only.
Same pattern as 23:57 trip script."""
import sys, json, base64, datetime
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

O, R = "robertyang87", "carbon-league"
out = {"trip": "2026-11-15 00:12 CST", "mark": "H23STATUS0012A"}

def branch_head(b):
    d = gh_api.req("GET", f"/repos/{O}/{R}/branches/{b}")
    c = d["commit"]
    return c["sha"][:8], c["commit"]["committer"]["date"], c["commit"]["message"].split("\n")[0]

out["branches"] = {}
for b in ["sync/h23", "sync/h29", "sync/h31", "main"]:
    try:
        sha, dt, msg = branch_head(b)
        out["branches"][b] = {"sha": sha, "date": dt, "msg": msg}
    except Exception as e:
        out["branches"][b] = {"error": str(e)[:120]}

def list_dir(path, ref):
    d = gh_api.get_file(O, R, path, ref=ref)
    return [d] if isinstance(d, dict) else d

try:
    out["synced_h23_items"] = len(list_dir("synced/h23", "sync/h23"))
except Exception as e:
    out["synced_h23_items"] = f"ERROR {e}"[:120]
try:
    out["bigeval_live_items"] = len(list_dir("synced/h23/bigeval_live", "sync/h23"))
except Exception as e:
    out["bigeval_live_items"] = f"ERROR {e}"[:120]

out["cal_out"] = {}
for name in ["m3s2r1pre_cal.out", "lineA_cal.out", "m7bsr1_cal.out"]:
    try:
        f = gh_api.get_file(O, R, f"synced/h23/{name}", ref="sync/h23")
        txt = base64.b64decode(f["content"]).decode(errors="replace")
        out["cal_out"][name] = {"size": f["size"], "text": txt}
    except Exception as e:
        out["cal_out"][name] = {"error": str(e)[:120]}

try:
    rl = gh_api.req("GET", "/rate_limit")
    core = rl["resources"]["core"]
    out["rate"] = {"remaining": core["remaining"],
                   "reset_cst": datetime.datetime.fromtimestamp(
                       core["reset"], datetime.timezone(datetime.timedelta(hours=8))).strftime("%H:%M CST")}
except Exception as e:
    out["rate"] = {"error": str(e)[:120]}

print(json.dumps(out, ensure_ascii=False, indent=2))
with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h23-status-last.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
