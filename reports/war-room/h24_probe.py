"""h24 sync-branch probe: head commit, dir listings, latest live file, cal.out, rate limit.
Usage: python3 h24_probe.py <tag>   (tag e.g. h24-0944; writes hidden_files/<tag>-probe.log)"""
import base64, hashlib, json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h24"
tag = sys.argv[1] if len(sys.argv) > 1 else "h24-probe"

def listing(path):
    return gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={BRANCH}")

def blob_text(path):
    d = gh_api.get_file(OWNER, REPO, path, ref=BRANCH)
    raw = base64.b64decode(d["content"])
    return d["sha"], hashlib.sha256(raw).hexdigest()[:8], raw.decode("utf-8", errors="replace")

out = {}
commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=1")
c = commits[0]
out["head"] = {"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"],
               "msg": c["commit"]["message"]}
d24 = listing("synced/h24")
out["synced/h24"] = {
    "count": len(d24),
    "cal_outs": [e["name"] for e in d24 if e["name"].endswith("_cal.out")],
    "names_tail": [e["name"] for e in d24[-5:]],
}
live = listing("synced/h24/bigeval_live")
out["synced/h24/bigeval_live"] = {
    "count": len(live),
    "cal_outs": [e["name"] for e in live if e["name"].endswith("_cal.out")],
    "names_tail": [e["name"] for e in live[-5:]],
}
sha, h, text = blob_text("synced/h24/bigeval_live/wt_BrU6S.jsonl")
lines = [l for l in text.splitlines() if l.strip()]
out["live_wt_BrU6S"] = {"lines": len(lines), "blob_sha": sha[:8], "sha256": h,
                        "last_line": lines[-1][:220] if lines else None}
sha2, h2, text2 = blob_text("synced/h24/m3s2r1_cal.out")
out["cal_out"] = {"blob_sha": sha2[:8], "sha256": h2, "text": text2}
rl = gh_api.req("GET", "/rate_limit")
core = rl["resources"]["core"]
out["rate_limit"] = {"remaining": core["remaining"], "limit": core["limit"],
                     "reset": core["reset"]}

path = f"/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/{tag}-probe.log"
with open(path, "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
print("WROTE", path)
