"""h23 status trip driver: direct read of sync/h23 branch (minimal API calls)."""
import json, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h23"

def main():
    out = {}
    # 1. rate limit (shared pool)
    try:
        rl = gh_api.req("GET", "/rate_limit")
        out["rate_limit"] = rl["resources"]["core"]
    except Exception as e:
        out["rate_limit_error"] = str(e)
    # 2. branch head
    br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{BR}")
    c = br["commit"]["commit"]
    out["branch_head"] = {"sha": br["commit"]["sha"][:8],
                          "date": c["committer"]["date"],
                          "message": c["message"][:120]}
    # 3. root listing of synced/h23
    root = gh_api.get_file(OWNER, REPO, "synced/h23", ref=BR)
    out["root_entries"] = [{"name": e["name"], "type": e["type"], "size": e.get("size")}
                           for e in root] if isinstance(root, list) else root
    # 4. bigeval_live listing
    live = gh_api.get_file(OWNER, REPO, "synced/h23/bigeval_live", ref=BR)
    if isinstance(live, list):
        names = sorted(e["name"] for e in live)
        out["live_count"] = len(names)
        out["live_newest_names"] = names[-15:]
        out["live_oldest_names"] = names[:5]
    else:
        out["live_listing"] = live
    # 5. cal.out files in root: fetch full text
    cal = {}
    for e in (root if isinstance(root, list) else []):
        if e["name"].endswith("_cal.out") or e["name"].endswith(".cal.out"):
            f = gh_api.get_file(OWNER, REPO, "synced/h23/" + e["name"], ref=BR)
            if isinstance(f, dict) and f.get("content"):
                cal[e["name"]] = base64.b64decode(f["content"]).decode("utf-8", "replace")
            else:
                cal[e["name"]] = f
    out["cal_out"] = cal
    print(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
