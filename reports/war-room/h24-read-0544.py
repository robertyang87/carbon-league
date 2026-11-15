"""H24 sync/h24 direct read probe h24-0544 (2026-11-15 ~05:44 CST).
Reads: recent commits on sync/h24, synced/h24 listing, bigeval_live listing
+ latest live file stats, *_cal.out full text, rate limit."""
import json, sys, base64, statistics
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h24"
out = {}
out["rate"] = gh_api.req("GET", "/rate_limit")["resources"]["core"]

commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=3")
out["commits"] = [
    {"sha": c["sha"][:7], "date": c["commit"]["committer"]["date"],
     "msg": c["commit"]["message"].splitlines()[0][:120]}
    for c in commits
]
listing = gh_api.get_file(OWNER, REPO, "synced/h24", ref=BRANCH)
out["h24_files"] = [e["name"] for e in listing] if isinstance(listing, list) else listing
live = gh_api.get_file(OWNER, REPO, "synced/h24/bigeval_live", ref=BRANCH)
out["live_count"] = len(live) if isinstance(live, list) else live
out["live_names_tail"] = [e["name"] for e in live][-5:] if isinstance(live, list) else None

def read_text(path):
    f = gh_api.get_file(OWNER, REPO, path, ref=BRANCH)
    if isinstance(f, dict) and "content" in f:
        return base64.b64decode(f["content"]).decode("utf-8", errors="replace")
    return None

def live_stats(name):
    txt = read_text(f"synced/h24/bigeval_live/{name}")
    if txt is None:
        return {"error": "unreadable"}
    lines = [l for l in txt.splitlines() if l.strip()]
    last = None
    for l in lines:
        try:
            last = json.loads(l)
        except Exception:
            pass
    return {"lines": len(lines), "sha": None, "last": last}

out["wt_BrU6S"] = live_stats("wt_BrU6S.jsonl")
for name in ["m3s2r1_cal.out", "m3s2r1_e0_cal.out"]:
    try:
        out[name] = read_text(f"synced/h24/{name}")
    except Exception as e:
        out[name] = f"ERROR {e}"

print(json.dumps(out, indent=1, ensure_ascii=False))
