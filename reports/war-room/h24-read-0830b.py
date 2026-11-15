"""h24 sync-branch direct read, run 2026-11-14 ~08:30 CST."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h24"
out = {}

br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{BRANCH}")
out["head_sha"] = br["commit"]["sha"]
out["head_date"] = br["commit"]["commit"]["committer"]["date"]
out["head_msg"] = br["commit"]["commit"]["message"]

def list_dir(path):
    items, page = [], 1
    while True:
        r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/{path}?ref={BRANCH}&per_page=100&page={page}")
        if not r:
            break
        items.extend(r)
        if len(r) < 100:
            break
        page += 1
    return items

top = list_dir("synced/h24")
out["top_count"] = len(top)
out["cal_outs"] = [i["name"] for i in top if i["name"].endswith("_cal.out")]

live = list_dir("synced/h24/bigeval_live")
out["live_count"] = len(live)

for name in out["cal_outs"]:
    f = gh_api.get_file(OWNER, REPO, f"synced/h24/{name}", ref=BRANCH)
    out.setdefault("cal_files", {})[name] = {
        "sha": f.get("sha"),
        "size": f.get("size"),
        "content": base64.b64decode(f["content"]).decode("utf-8", errors="replace"),
    }

print(json.dumps(out, indent=2, ensure_ascii=False))
with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24-read-0830b.json", "w") as fh:
    json.dump(out, fh, indent=2, ensure_ascii=False)
