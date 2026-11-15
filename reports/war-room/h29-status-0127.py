"""H23 status trip 01:27 - direct read of sync/h29 branch (spot-29 backfill).
Tag: H29STATUS0127. Read-only; prints branch head, live-file line counts,
seed-domain check, *_cal.out contents, and sync/h23 head for expected-silence
confirmation. No writes, no dispatches."""
import base64, json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
BRANCH = "sync/h29"

def b64decode(content):
    return base64.b64decode("".join(content.split())).decode("utf-8", errors="replace")

def get_text(path, ref):
    r = gh_api.get_file(OWNER, REPO, path, ref=ref)
    if isinstance(r, dict) and r.get("content"):
        return b64decode(r["content"])
    return None

def list_dir(path, ref):
    try:
        r = gh_api.get_file(OWNER, REPO, path, ref=ref)
    except Exception as e:
        return {"error": str(e)[:200]}
    if isinstance(r, list):
        return [(e["name"], e.get("size"), e.get("type")) for e in r]
    return {"error": f"unexpected: {str(r)[:200]}"}

out = {}
# 0. rate limit (quota economy check)
rl = gh_api.req("GET", "/rate_limit")
out["rate_core"] = rl["resources"]["core"]

# 1. sync/h29 branch head
br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{BRANCH}")
out["h29_head_sha"] = br["commit"]["sha"]
out["h29_head_date"] = br["commit"]["commit"]["author"]["date"]
out["h29_head_msg"] = br["commit"]["commit"]["message"][:160]

# 2. live dir listing + 8 files
live = list_dir("synced/h29/bigeval_live", BRANCH)
out["live_listing"] = live
out["live_files"] = {}
total = 0
if isinstance(live, list):
    for name, size, _t in live:
        if not name.endswith(".jsonl"):
            continue
        txt = get_text(f"synced/h29/bigeval_live/{name}", BRANCH)
        if txt is None:
            out["live_files"][name] = {"error": "fetch failed/empty"}
            continue
        lines = [l for l in txt.splitlines() if l.strip()]
        seeds = []
        bad = []
        for l in lines:
            try:
                s = json.loads(l).get("seed")
                if s is not None:
                    seeds.append(int(s))
            except Exception:
                pass
        # domain: _m2 -> 996000-996199, _echo2 -> 999000-999199
        if name.endswith("_m2.jsonl"):
            bad = [s for s in seeds if not 996000 <= s < 996200]
        elif name.endswith("_echo2.jsonl"):
            bad = [s for s in seeds if not 999000 <= s < 999200]
        total += len(lines)
        out["live_files"][name] = {
            "lines": len(lines), "size": size,
            "seeds_n": len(seeds),
            "min_seed": min(seeds) if seeds else None,
            "max_seed": max(seeds) if seeds else None,
            "unique_seeds": len(set(seeds)),
            "bad_seeds": bad,
        }
out["total_lines"] = total

# 3. synced/h29 root listing (cal.out files)
root = list_dir("synced/h29", BRANCH)
out["root_listing"] = root
out["cal_outs"] = {}
if isinstance(root, list):
    for name, size, _t in root:
        if name.endswith("_cal.out") or name.endswith(".out"):
            txt = get_text(f"synced/h29/{name}", BRANCH)
            out["cal_outs"][name] = txt[:2000] if txt else None

# 4. sync/h23 head (expected silent)
br23 = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/sync/h23")
out["h23_head_sha"] = br23["commit"]["sha"]
out["h23_head_date"] = br23["commit"]["commit"]["author"]["date"]
out["h23_head_msg"] = br23["commit"]["commit"]["message"][:120]

print(json.dumps(out, indent=1, ensure_ascii=False))
