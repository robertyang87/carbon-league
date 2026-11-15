"""H24 00:29 read 2026-11-14: sync/h24 + sync/h31 + sync/h29 + main heads,
top listings, bigeval_live counts, m3s2r1_cal.out full read, rate limit."""
import base64, json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
HF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/"
out = {"run": "h24-status-report 00:29 CST 2026-11-14"}

def head(br):
    commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={br}&per_page=1")
    c = commits[0]
    return {"sha": c["sha"][:8], "date": c["commit"]["committer"]["date"],
            "msg": c["commit"]["message"].splitlines()[0][:140]}

def listing(path, ref):
    try:
        r = gh_api.get_file(OWNER, REPO, path, ref=ref)
        if isinstance(r, list):
            return [{"name": e["name"], "size": e.get("size"), "sha": e.get("sha", "")[:8]} for e in r]
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
h29_live = listing("synced/h29/bigeval_live", "sync/h29")

out["h24_top_count"] = len(h24_top) if isinstance(h24_top, list) else h24_top
out["h24_out_count"] = len([e for e in h24_top if isinstance(e, dict) and e["name"].endswith(".out")]) if isinstance(h24_top, list) else h24_top
out["h24_cal_outs"] = [e["name"] for e in h24_top if isinstance(e, dict) and e["name"].endswith("_cal.out")] if isinstance(h24_top, list) else h24_top
out["h24_live_count"] = len(h24_live) if isinstance(h24_live, list) else h24_live
out["h31_top_count"] = len(h31_top) if isinstance(h31_top, list) else h31_top
out["h31_live_count"] = len(h31_live) if isinstance(h31_live, list) else h31_live
out["h31_live_names"] = [e["name"] for e in h31_live] if isinstance(h31_live, list) else h31_live
out["h29_top_count"] = len(h29_top) if isinstance(h29_top, list) else h29_top
out["h29_live_count"] = len(h29_live) if isinstance(h29_live, list) else h29_live

# m3s2r1_cal.out full read (verify content unchanged vs 00:14 report)
try:
    f = gh_api.get_file(OWNER, REPO, "synced/h24/m3s2r1_cal.out", ref="sync/h24")
    content = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
    out["cal_out_sha"] = f["sha"][:8]
    out["cal_out_size"] = f["size"]
    out["cal_out_text"] = content
except Exception as e:
    out["cal_out_error"] = str(e)[:300]

try:
    rl = gh_api.req("GET", "/rate_limit")
    out["rate_limit_core_remaining"] = rl["resources"]["core"]["remaining"]
    out["rate_limit_core_reset"] = rl["resources"]["core"]["reset"]
except Exception as e:
    out["rate_limit_error"] = str(e)[:200]

print(json.dumps(out, indent=1, ensure_ascii=False))
with open(HF + "h24-read-0029.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
