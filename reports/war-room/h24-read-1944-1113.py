"""H24 19:44 read 2026-11-13: sync/h24 + sync/h31 branch heads, synced top
listings, bigeval_live listings, snap_70 live files line counts (expect
200/200 complete), any new *_cal.out on h24/h31."""
import base64, json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {"run": "h24-status-report 19:44 CST 2026-11-13"}
for br in ["sync/h31", "sync/h24"]:
    commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={br}&per_page=3")
    out[br] = [
        {"sha": c["sha"][:7], "date": c["commit"]["committer"]["date"],
         "msg": c["commit"]["message"].splitlines()[0][:140]}
        for c in commits
    ]

def listing(path, ref):
    try:
        r = gh_api.get_file(OWNER, REPO, path, ref=ref)
        if isinstance(r, list):
            return [{"name": e["name"], "size": e.get("size"), "sha": e.get("sha", "")[:7]} for e in r]
        return {"not_a_listing": str(r)[:200]}
    except Exception as e:
        return {"error": str(e)[:300]}

out["h31_top"] = listing("synced/h31", "sync/h31")
out["h31_live"] = listing("synced/h31/bigeval_live", "sync/h31")
out["h24_top"] = listing("synced/h24", "sync/h24")
h24_live = listing("synced/h24/bigeval_live", "sync/h24")
if isinstance(h24_live, list):
    out["h24_live_count"] = len(h24_live)
    out["h24_live_sample"] = h24_live[:3]
else:
    out["h24_live"] = h24_live

def live_lines(branch, name):
    try:
        f = gh_api.get_file(OWNER, REPO, f"synced/{branch}/bigeval_live/{name}", ref=f"sync/{branch}")
        raw = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
        lines = [l for l in raw.splitlines() if l.strip()]
        return {"lines": len(lines), "size": len(raw),
                "last_line": lines[-1][:300] if lines else None}
    except Exception as e:
        return {"error": str(e)[:300]}

if isinstance(out["h31_live"], list):
    for name in ("snap_70_m2.jsonl", "snap_70_echo2.jsonl"):
        hit = [e for e in out["h31_live"] if isinstance(e, dict) and e.get("name") == name]
        out[name] = live_lines("h31", name) if hit else {"absent": True}

if isinstance(out["h31_top"], list):
    out["h31_cal_outs"] = [e for e in out["h31_top"] if e["name"].endswith("_cal.out")]
if isinstance(out["h24_top"], list):
    out["h24_cal_outs"] = [e for e in out["h24_top"] if e["name"].endswith("_cal.out")]
print(json.dumps(out, indent=1, ensure_ascii=False))
with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24-read-1944-1113.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
