#!/usr/bin/env python3
"""h24-1059 probe: direct read of sync/h24 branch (GitHub API)."""
import sys, json, base64, hashlib
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h24"
out = {}

br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{BR}")
out["head"] = {"sha": br["commit"]["sha"][:8],
               "date": br["commit"]["commit"]["committer"]["date"],
               "msg": br["commit"]["commit"]["message"]}

def list_dir(path):
    r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={BR}")
    return r

d1 = list_dir("synced/h24")
names1 = sorted(e["name"] for e in d1)
out["synced/h24"] = {"count": len(names1),
                     "cal_outs": [n for n in names1 if n.endswith("_cal.out")],
                     "names_tail": names1[-5:]}
d2 = list_dir("synced/h24/bigeval_live")
names2 = sorted(e["name"] for e in d2)
out["synced/h24/bigeval_live"] = {"count": len(names2),
                                  "cal_outs": [n for n in names2 if n.endswith("_cal.out")],
                                  "names_tail": names2[-5:]}

live = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/synced/h24/bigeval_live/wt_BrU6S.jsonl?ref={BR}")
raw = base64.b64decode(live["content"])
lines = raw.decode().splitlines()
out["live_wt_BrU6S"] = {"lines": len(lines), "blob_sha": live["sha"][:8],
                        "sha256": hashlib.sha256(raw).hexdigest()[:8],
                        "last_line": lines[-1][:400] if lines else None}

cal = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/synced/h24/m3s2r1_cal.out?ref={BR}")
craw = base64.b64decode(cal["content"])
out["cal_out"] = {"blob_sha": cal["sha"][:8],
                  "sha256": hashlib.sha256(craw).hexdigest()[:8],
                  "text": craw.decode()}

rl = gh_api.req("GET", "/rate_limit")
core = rl["resources"]["core"]
out["rate_limit"] = {"remaining": core["remaining"], "limit": core["limit"], "reset": core["reset"]}

p = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24-1059-probe.log"
with open(p, "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False)[:3000])
print("WROTE", p)
