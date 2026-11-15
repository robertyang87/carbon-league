"""H24 19:14 read: sync/h24 + sync/h31 branch heads, synced/h31 + synced/h24
top listings, bigeval_live listings with sizes. it570 g1 exam started on
spot-31 at 19:12:38 CST (pid 20226); look for new m9xj570g1_* live file."""
import base64, json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {"run": "h24-status-report 19:14 CST 2026-11-13"}
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

# New it570 g1 live file, if present: count lines + last line
for name in ("m9xj570g1_m2.jsonl", "m9xj570g1_echo2.jsonl"):
    hit = [e for e in out["h31_live"] if isinstance(e, dict) and e.get("name") == name]
    if hit:
        try:
            f = gh_api.get_file(OWNER, REPO, f"synced/h31/bigeval_live/{name}", ref="sync/h31")
            raw = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
            lines = [l for l in raw.splitlines() if l.strip()]
            out[name] = {"lines": len(lines), "size": len(raw),
                         "last_line": lines[-1][:300] if lines else None}
        except Exception as e:
            out[name] = {"error": str(e)[:300]}
print(json.dumps(out, indent=1, ensure_ascii=False))
