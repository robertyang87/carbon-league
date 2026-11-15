"""H24 21:59 read 2026-11-13: sync/h24 + sync/h31 + sync/h29 + main heads,
top listings, bigeval_live counts, *_cal.out names."""
import json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
HF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/"
out = {"run": "h24-status-report 21:59 CST 2026-11-13"}

def head(br):
    commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={br}&per_page=1")
    c = commits[0]
    return {"sha": c["sha"][:7], "date": c["commit"]["committer"]["date"],
            "msg": c["commit"]["message"].splitlines()[0][:140]}

def listing(path, ref):
    try:
        r = gh_api.get_file(OWNER, REPO, path, ref=ref)
        if isinstance(r, list):
            return [{"name": e["name"], "size": e.get("size")} for e in r]
        return {"not_a_listing": str(r)[:200]}
    except Exception as e:
        return {"error": str(e)[:300]}

for br in ["sync/h24", "sync/h31", "sync/h29", "main"]:
    try:
        out[br] = head(br)
    except Exception as e:
        out[br] = {"error": str(e)[:300]}

h24_top = listing("synced/h24", "sync/h24")
h24_live = listing("synced/h24/bigeval_live", "sync/h24")
h31_top = listing("synced/h31", "sync/h31")
h31_live = listing("synced/h31/bigeval_live", "sync/h31")
h29_top = listing("synced/h29", "sync/h29")

out["h24_top_count"] = len(h24_top) if isinstance(h24_top, list) else h24_top
out["h24_cal_outs"] = [e["name"] for e in h24_top if isinstance(e, dict) and e["name"].endswith("_cal.out")] if isinstance(h24_top, list) else h24_top
out["h24_live_count"] = len(h24_live) if isinstance(h24_live, list) else h24_live
out["h31_top_count"] = len(h31_top) if isinstance(h31_top, list) else h31_top
out["h31_live_count"] = len(h31_live) if isinstance(h31_live, list) else h31_live
out["h31_live_names"] = [e["name"] for e in h31_live] if isinstance(h31_live, list) else h31_live
out["h29_top_count"] = len(h29_top) if isinstance(h29_top, list) else h29_top

print(json.dumps(out, indent=1, ensure_ascii=False))
with open(HF + "h24-read-2159.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
