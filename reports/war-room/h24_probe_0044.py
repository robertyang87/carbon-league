"""h24 status probe — marker h24-0044 — GitHub API direct read of sync/h24."""
import json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h24"
out = {"marker": "h24-0044"}

# 1. branch head
br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{BRANCH}")
head = br["commit"]
out["branch_head_sha"] = head["sha"][:8]
out["branch_head_full"] = head["sha"]
commit = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits/{head['sha']}")
out["branch_head_time"] = commit["commit"]["committer"]["date"]
out["branch_head_msg"] = commit["commit"]["message"][:120]

# 2. synced/h24 listing
def listing(path):
    d = gh_api.get_file(OWNER, REPO, path, ref=BRANCH)
    if isinstance(d, dict):
        return [d]
    return d

h24 = listing("synced/h24")
out["synced_h24_count"] = len(h24)
out["cal_files"] = sorted(
    [e["name"] for e in h24 if e["name"].endswith("_cal.out")])
live = listing("synced/h24/bigeval_live")
out["bigeval_live_count"] = len(live)
out["live_files"] = {e["name"]: {"size": e["size"], "sha": e["sha"][:8]}
                     for e in live if e["name"].startswith("m3s2r1")}

# 3. cal.out full text (only file)
for name in out["cal_files"]:
    f = gh_api.get_file(OWNER, REPO, f"synced/h24/{name}", ref=BRANCH)
    import base64
    out[f"cal_{name}"] = {
        "size": f["size"], "sha": f["sha"][:8],
        "text": base64.b64decode(f["content"]).decode("utf-8", errors="replace"),
    }

print(json.dumps(out, ensure_ascii=False, indent=1))
