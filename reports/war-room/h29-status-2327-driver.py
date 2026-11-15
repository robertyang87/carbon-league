"""H23/C1 status trip 23:27 - direct read of sync/h29 branch (spot-29).
Tag: H23STATUS2327. Read-only; prints branch head, live-file line counts,
last-game times, and *_cal.out contents. No writes, no dispatches."""
import base64, json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h29"

def b64decode(content):
    return base64.b64decode("".join(content.split())).decode("utf-8", errors="replace")

def get_text(path):
    r = gh_api.get_file(OWNER, REPO, path, ref=BRANCH)
    if isinstance(r, dict) and r.get("content"):
        return b64decode(r["content"])
    return None

def list_dir(path):
    try:
        r = gh_api.get_file(OWNER, REPO, path, ref=BRANCH)
    except Exception as e:
        return {"error": str(e)[:200]}
    if isinstance(r, list):
        return [(e["name"], e.get("size"), e.get("type")) for e in r]
    return {"error": f"unexpected: {str(r)[:200]}"}

out = {}
# 1. branch head
br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{BRANCH}")
out["branch_head_sha"] = br["commit"]["sha"]
out["branch_head_date"] = br["commit"]["commit"]["author"]["date"]
out["branch_head_msg"] = br["commit"]["commit"]["message"][:120]

# 2. live dir listing
live = list_dir("synced/h29/bigeval_live")
out["live_listing"] = live
out["live_files"] = {}
if isinstance(live, list):
    for name, size, _t in live:
        if not name.endswith(".jsonl"):
            continue
        txt = get_text(f"synced/h29/bigeval_live/{name}")
        if txt is None:
            out["live_files"][name] = {"error": "fetch failed/empty"}
            continue
        lines = [l for l in txt.splitlines() if l.strip()]
        last = None
        try:
            last = json.loads(lines[-1]) if lines else None
        except Exception:
            last = {"raw_tail": lines[-1][:200] if lines else None}
        out["live_files"][name] = {"lines": len(lines), "size": size, "last": last}

# 3. synced/h29 root listing (cal.out files)
root = list_dir("synced/h29")
out["root_listing"] = root
out["cal_outs"] = {}
if isinstance(root, list):
    for name, size, _t in root:
        if name.endswith("_cal.out") or name.endswith(".out"):
            txt = get_text(f"synced/h29/{name}")
            out["cal_outs"][name] = {"size": size, "text": txt}

# 4. bigeval.jsonl summaries (small file expected)
bj = get_text("synced/h29/bigeval.jsonl")
out["bigeval_jsonl_tail"] = bj.splitlines()[-6:] if bj else None

print(json.dumps(out, indent=1, ensure_ascii=False))
