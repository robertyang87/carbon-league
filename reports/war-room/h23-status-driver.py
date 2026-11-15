#!/usr/bin/env python3
"""h23 status direct read via gh_api.py (sync/h23 branch)."""
import sys, json, hashlib
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h23"

def contents(path):
    r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={BR}")
    return r

def get_raw(path):
    # get_file returns parsed json for contents API; use req for raw content
    r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={BR}")
    return r

out = {}
# branch head
br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{BR}")
c = br["commit"]["commit"]
out["branch"] = {"sha": br["commit"]["sha"][:8], "date": c["committer"]["date"],
                 "msg": c["message"]}
# rate limit
rl = gh_api.req("GET", "/rate_limit")
core = rl["resources"]["core"]
out["rate"] = {"remaining": core["remaining"],
               "reset_cst": core["reset"]}
# top level synced/h23
top = contents("synced/h23")
names = sorted(e["name"] for e in top)
out["synced_h23_top_level_items"] = len(top)
out["synced_h23_top_level_names_md5"] = hashlib.md5("\n".join(names).encode()).hexdigest()
# subdirs
subs = {}
for e in top:
    if e["type"] == "dir":
        sub = contents(f"synced/h23/{e['name']}")
        subs[e["name"]] = len(sub)
out["synced_h23_subdirs"] = subs
# bigeval_live
live = contents("synced/h23/bigeval_live")
out["bigeval_live_items"] = len(live)
live_names = sorted(e["name"] for e in live)
out["bigeval_live_last"] = live_names[-5:]
tails = {}
for name in live_names[-5:]:
    f = get_raw(f"synced/h23/bigeval_live/{name}")
    import base64
    raw = base64.b64decode(f["content"]).decode("utf-8", "replace")
    lines = [l for l in raw.splitlines() if l.strip()]
    last = None
    if lines:
        try:
            d = json.loads(lines[-1])
            last = {"margin": d.get("margin"), "seed": d.get("seed")}
        except Exception:
            last = {"raw_tail": lines[-1][:120]}
    tails[name] = {"lines": len(lines), "last": last}
out["bigeval_live_tails"] = tails
# cal.out files
cal = {}
for name in ["lineA_cal.out", "m3s2r1pre_cal.out", "m7bsr1_cal.out"]:
    try:
        f = get_raw(f"synced/h23/{name}")
        import base64
        raw = base64.b64decode(f["content"]).decode("utf-8", "replace")
        cal[name] = {"size": len(raw), "text": raw}
    except Exception as ex:
        cal[name] = {"error": str(ex)}
out["cal_out"] = cal
print(json.dumps(out, ensure_ascii=False, indent=1))
