"""H24 20:14 read 2026-11-13: sync/h24 + sync/h31 + main heads, top listings,
bigeval_live line counts for completed exams, *_cal.out contents, summaries dir."""
import base64, json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
HF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/"
out = {"run": "h24-status-report 20:14 CST 2026-11-13"}
for br in ["sync/h31", "sync/h24", "main"]:
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

def get_text(branch, path):
    try:
        f = gh_api.get_file(OWNER, REPO, path, ref=f"sync/{branch}")
        raw = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
        return raw
    except Exception as e:
        return f"ERROR: {e}"[:300]

out["h31_top"] = listing("synced/h31", "sync/h31")
out["h31_live"] = listing("synced/h31/bigeval_live", "sync/h31")
out["h31_summaries"] = listing("synced/h31/summaries", "sync/h31")
out["h24_top"] = listing("synced/h24", "sync/h24")
h24_live = listing("synced/h24/bigeval_live", "sync/h24")
out["h24_live_count"] = len(h24_live) if isinstance(h24_live, list) else h24_live

def live_lines(branch, name):
    raw = get_text(branch, f"synced/{branch}/bigeval_live/{name}")
    if raw.startswith("ERROR"):
        return {"error": raw}
    lines = [l for l in raw.splitlines() if l.strip()]
    return {"lines": len(lines), "size": len(raw),
            "last_line": lines[-1][:300] if lines else None}

if isinstance(out["h31_live"], list):
    for name in ("snap_70_echo2.jsonl", "it570g1_m2.jsonl", "it570g1_echo2.jsonl"):
        hit = [e for e in out["h31_live"] if isinstance(e, dict) and e.get("name") == name]
        out[name] = live_lines("h31", name) if hit else {"absent": True}

if isinstance(out["h31_top"], list):
    out["h31_cal_outs"] = [e for e in out["h31_top"] if e["name"].endswith("_cal.out")]
    for e in out["h31_cal_outs"]:
        out[f"cal:{e['name']}"] = get_text("h31", f"synced/h31/{e['name']}")[:2000]
if isinstance(out["h24_top"], list):
    out["h24_cal_outs"] = [e for e in out["h24_top"] if e["name"].endswith("_cal.out")]

print(json.dumps(out, indent=1, ensure_ascii=False))
with open(HF + "h24-read-2014.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
