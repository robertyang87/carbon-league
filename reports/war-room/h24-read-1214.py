"""H24 12:14 trip direct read. sync/h24 + sync/h30 heads, dir listings.
Minimal API calls (shared quota)."""
import json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {}

for br in ["sync/h24", "sync/h30"]:
    commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={br}&per_page=3")
    out[br] = [
        {"sha": c["sha"][:7], "date": c["commit"]["committer"]["date"],
         "msg": c["commit"]["message"].splitlines()[0][:120]}
        for c in commits
    ]

for br, path in [("sync/h24", "synced/h24"),
                 ("sync/h24", "synced/h24/bigeval_live"),
                 ("sync/h30", "synced/h30"),
                 ("sync/h30", "synced/h30/bigeval_live")]:
    try:
        listing = gh_api.get_file(OWNER, REPO, path, ref=br)
        key = f"{br}:{path}"
        if isinstance(listing, list):
            out[key] = [{"name": e["name"], "size": e.get("size")} for e in listing]
        else:
            out[key] = listing
    except Exception as e:
        out[f"{br}:{path}"] = {"error": str(e)}

print(json.dumps(out, indent=1, ensure_ascii=False))
