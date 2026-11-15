import sys, json, hashlib, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h24"

def get_json(path):
    r = gh_api.req("GET", path)
    assert r is not None, path
    return r

out = {}

br = get_json(f"/repos/{OWNER}/{REPO}/branches/{BR}")
c = br["commit"]["commit"]
out["head"] = {"sha": br["commit"]["sha"][:8],
               "date": c["committer"]["date"],
               "msg": c["message"]}

def list_dir(path):
    items = get_json(f"/repos/{OWNER}/{REPO}/contents/{path}?ref={BR}")
    return items

d1 = list_dir("synced/h24")
out["synced/h24"] = {"count": len(d1),
                     "cal_outs": sorted(i["name"] for i in d1 if i["name"].endswith("_cal.out")),
                     "names_tail": [i["name"] for i in d1[-5:]]}
d2 = list_dir("synced/h24/bigeval_live")
out["synced/h24/bigeval_live"] = {"count": len(d2),
                                  "cal_outs": sorted(i["name"] for i in d2 if i["name"].endswith("_cal.out")),
                                  "names_tail": [i["name"] for i in d2[-5:]]}

f = get_json(f"/repos/{OWNER}/{REPO}/contents/synced/h24/bigeval_live/wt_BrU6S.jsonl?ref={BR}")
raw = base64.b64decode(f["content"])
lines = raw.decode().strip().splitlines()
out["live_wt_BrU6S"] = {"lines": len(lines),
                        "blob_sha": f["sha"][:8],
                        "sha256": hashlib.sha256(raw).hexdigest()[:8],
                        "last_line": lines[-1][:400] if lines else None}

g = get_json(f"/repos/{OWNER}/{REPO}/contents/synced/h24/m3s2r1_cal.out?ref={BR}")
raw2 = base64.b64decode(g["content"])
out["cal_out"] = {"blob_sha": g["sha"][:8],
                  "sha256": hashlib.sha256(raw2).hexdigest()[:8],
                  "text": raw2.decode()}

rl = get_json("/rate_limit")
out["rate_limit"] = {"remaining": rl["resources"]["core"]["remaining"],
                     "limit": rl["resources"]["core"]["limit"],
                     "reset": rl["resources"]["core"]["reset"]}

print(json.dumps(out, indent=1, ensure_ascii=False))
with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24-1114-probe.log", "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
