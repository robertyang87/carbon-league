"""H24 15:14 run: direct read of sync/h31 (it320 chain, spot-31) + sync/h24 (spot-24, expected silent).
Reads: branch heads, synced/h31 tree (sizes), it320 live files line counts + last game time + wins/mean recompute, *_cal.out full text.
"""
import json, sys, base64, statistics
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {"run": "h24-status-report 15:14 CST 2026-11-13"}

def get(path):
    return gh_api.req("GET", path)

# 1. branch heads
for br in ["sync/h31", "sync/h24"]:
    commits = get(f"/repos/{OWNER}/{REPO}/commits?sha={br}&per_page=3")
    out[br] = [{"sha": c["sha"][:7], "date": c["commit"]["committer"]["date"],
                "msg": c["commit"]["message"].splitlines()[0][:100]} for c in commits]

# 2. tree of synced/h31 (recursive, to get sizes without fetching contents)
tree = get(f"/repos/{OWNER}/{REPO}/git/trees/sync/h31?recursive=1")
h31 = [t for t in tree.get("tree", []) if t["path"].startswith("synced/h31/")]
out["h31_files"] = [{"path": t["path"][len("synced/h31/"):], "size": t.get("size")} for t in h31]

# 3. it320 live files: line count, wins, mean, last game time/seed
def read_text(path, ref):
    f = gh_api.get_file(OWNER, REPO, path, ref=ref)
    if isinstance(f, dict) and "content" in f:
        return base64.b64decode(f["content"]).decode("utf-8", errors="replace")
    return None

def live_stats(name):
    p = f"synced/h31/bigeval_live/{name}"
    txt = read_text(p, "sync/h31")
    if txt is None:
        return {"error": "unreadable (contents API may refuse >1MB)"}
    lines = [l for l in txt.splitlines() if l.strip()]
    wins = 0; margins = []; last = None; seeds = set()
    for l in lines:
        try:
            j = json.loads(l)
        except Exception:
            continue
        last = j
        m = j.get("margin", j.get("score_diff"))
        if isinstance(m, (int, float)):
            margins.append(m)
            if m > 0:
                wins += 1
        s = j.get("seed")
        if isinstance(s, int):
            seeds.add(s)
    return {"lines": len(lines), "wins": wins,
            "mean": round(statistics.mean(margins), 1) if margins else None,
            "se": round(statistics.stdev(margins) / (len(margins) ** 0.5), 1) if len(margins) > 1 else None,
            "worst": min(margins) if margins else None,
            "seeds_unique": len(seeds),
            "last_seed": last.get("seed") if isinstance(last, dict) else None,
            "last_keys": sorted(last.keys()) if isinstance(last, dict) else None}

for name in ["m9xj320r1_m2.jsonl", "m9xj320r1_echo2.jsonl",
             "m9xj320g1_m2.jsonl", "m9xj320g1_echo2.jsonl",
             "m9xj320wr1_m2.jsonl", "m9xj320wr1_echo2.jsonl"]:
    try:
        out[name] = live_stats(name)
    except Exception as e:
        out[name] = {"error": str(e)}

# 4. cal.out files under synced/h31
for name in ["m9xj320r1_cal.out", "m9xj320g1_cal.out", "m9xj320wr1_cal.out",
             "k4v2r1_cal.out"]:
    p = f"synced/h31/{name}"
    try:
        txt = read_text(p, "sync/h31")
        out[name] = txt if txt is not None else "MISSING"
    except Exception as e:
        out[name] = f"ERROR {e}"

print(json.dumps(out, indent=1, ensure_ascii=False))
