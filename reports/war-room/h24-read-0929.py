"""H24 09:29 status read driver. sync/h24 head, recursive tree counts,
full text of any *_cal.out under synced/h24."""
import base64, json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {"run_tag": "h24-0929", "checked_at_cst": "2026-11-14 09:29-09:32"}

try:
    commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha=sync/h24&per_page=1")
    c = commits[0]
    out["head_sha"] = c["sha"]
    out["head_date"] = c["commit"]["committer"]["date"]
    out["head_msg"] = c["commit"]["message"].splitlines()[0][:120]
except Exception as e:
    out["head_error"] = repr(e)

try:
    tree = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/git/trees/{out['head_sha']}?recursive=1")
    paths = [t["path"] for t in tree.get("tree", [])]
    synced = [p for p in paths if p.startswith("synced/h24/")]
    top = [p for p in synced if p.count("/") == 2]
    live = [p for p in synced if p.startswith("synced/h24/bigeval_live/")]
    out["top_count"] = len(top)
    out["live_count"] = len(live)
    out["cal_outs"] = sorted(p.split("/")[-1] for p in synced if p.endswith("_cal.out"))
    out["tree_truncated"] = tree.get("truncated")
except Exception as e:
    out["tree_error"] = repr(e)

out["cal_text"] = {}
for name in out.get("cal_outs", []):
    try:
        f = gh_api.get_file(OWNER, REPO, f"synced/h24/{name}", ref="sync/h24")
        out["cal_text"][name] = {
            "sha": f.get("sha"),
            "size": f.get("size"),
            "text": base64.b64decode(f["content"]).decode("utf-8", errors="replace"),
        }
    except Exception as e:
        out["cal_text"][name] = {"error": repr(e)}

print(json.dumps(out, indent=1, ensure_ascii=False))
with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24-read-0929.json", "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
