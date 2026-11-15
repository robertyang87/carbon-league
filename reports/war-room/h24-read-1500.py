"""H24 15:00 read: sync/h31 + sync/h24 branch heads, synced/h31 listing,
bigeval_live listing. (Exam it320 runs on spot-31; sync/h31 is manual-push
after completion per war-room confirmed口径 — live progress comes from
machine direct read, not this branch.)"""
import json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {}
for br in ["sync/h31", "sync/h24"]:
    commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={br}&per_page=3")
    out[br] = [
        {"sha": c["sha"][:7], "date": c["commit"]["committer"]["date"],
         "msg": c["commit"]["message"].splitlines()[0][:140]}
        for c in commits
    ]
try:
    live = gh_api.get_file(OWNER, REPO, "synced/h31/bigeval_live", ref="sync/h31")
    out["h31_live_files"] = [e["name"] for e in live] if isinstance(live, list) else live
except Exception as e:
    out["h31_live_files_error"] = str(e)
try:
    top = gh_api.get_file(OWNER, REPO, "synced/h31", ref="sync/h31")
    out["h31_top_files"] = [e["name"] for e in top] if isinstance(top, list) else top
except Exception as e:
    out["h31_top_files_error"] = str(e)
print(json.dumps(out, indent=1, ensure_ascii=False))
