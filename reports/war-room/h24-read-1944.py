"""h24 direct read via GitHub API (sync/h24 branch). Run: python3 h24-read-1944.py"""
import base64, json, sys
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req, get_file

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h24"
out = {}

commits = req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=8")
out["commits"] = [
    {"sha": c["sha"][:7], "date": c["commit"]["committer"]["date"],
     "msg": c["commit"]["message"][:80]} for c in commits]

def listing(path):
    r = get_file(OWNER, REPO, path, ref=BRANCH)
    if isinstance(r, list):
        return [e["name"] for e in r]
    return {"error": str(r)[:200]}

out["h24_files"] = listing("synced/h24")
out["live_files"] = listing("synced/h24/bigeval_live")

def file_text(path):
    r = get_file(OWNER, REPO, path, ref=BRANCH)
    if isinstance(r, dict) and r.get("content"):
        return base64.b64decode(r["content"]).decode("utf-8", errors="replace")
    return f"UNREADABLE: {str(r)[:200]}"

out["m3s2r1_cal.out"] = file_text("synced/h24/m3s2r1_cal.out")

# live files: tail each jsonl, count lines, last line
live = {}
for name in out["live_files"] if isinstance(out["live_files"], list) else []:
    if name.endswith(".jsonl"):
        txt = file_text(f"synced/h24/bigeval_live/{name}")
        lines = [l for l in txt.splitlines() if l.strip()]
        live[name] = {"n_lines": len(lines),
                      "last_line": lines[-1][:400] if lines else None,
                      "first_line": lines[0][:200] if lines else None}
out["live"] = live

print(json.dumps(out, indent=1, ensure_ascii=False))
with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24-read-1944.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
