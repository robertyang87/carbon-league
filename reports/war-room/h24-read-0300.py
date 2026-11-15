"""H24 status read, trip 03:00 CST 2026-11-14. Direct GitHub API read of
robertyang87/carbon-league sync/h24 (+ main head for bigeval-fix context).
Writes h24-read-0300.json next to this script."""
import base64, json, os, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h24-read-0300.json")
OWNER, REPO = "robertyang87", "carbon-league"

def branch_head(branch):
    b = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{branch}")
    c = b["commit"]["commit"]
    return {"sha": b["commit"]["sha"][:7], "date": c["committer"]["date"],
            "message": c["message"].splitlines()[0][:120]}

res = {"trip": "h24-status-report 03:00", "read_at_cst": "2026-11-14 03:00"}
try:
    res["rate_limit"] = gh_api.req("GET", "/rate_limit")["resources"]["core"]
except Exception as e:
    res["rate_limit"] = {"error": str(e)}

res["sync_h24_head"] = branch_head("sync/h24")
res["main_head"] = branch_head("main")
res["sync_h31_head"] = branch_head("sync/h31")

# Recursive tree of sync/h24 head -> counts for synced/h24 and bigeval_live
tree = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/git/trees/{res['sync_h24_head']['sha']}?recursive=1")
paths = [t["path"] for t in tree.get("tree", [])]
h24 = [p for p in paths if p.startswith("synced/h24/")]
live = [p for p in h24 if p.startswith("synced/h24/bigeval_live/")]
res["synced_h24_items"] = len(h24)
res["bigeval_live_items"] = len(live)
res["cal_out_files"] = sorted(p for p in h24 if p.endswith("_cal.out"))

# Full text of the cal.out file(s) — settle the vs-Echo2 144W/+472 vs 146W/+409 question
res["cal_out"] = {}
for p in res["cal_out_files"]:
    f = gh_api.get_file(OWNER, REPO, p, ref="sync/h24")
    content = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
    res["cal_out"][p] = {"sha": f["sha"][:10], "size": f["size"], "text": content}

with open(OUT, "w") as fh:
    json.dump(res, fh, indent=2, ensure_ascii=False)
print(json.dumps(res, indent=2, ensure_ascii=False)[:6000])
