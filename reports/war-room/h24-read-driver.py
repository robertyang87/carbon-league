"""H24 sync/h24 direct read driver. Run: python3 h24-read-driver.py
Reads: recent commits on sync/h24, synced/h24 dir listing, bigeval_live jsonl
line counts + last game time, *_cal.out full text."""
import json, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h24"

def get(path):
    return gh_api.req("GET", path)

out = {}
# 1. recent commits on sync/h24
commits = get(f"/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=6")
out["commits"] = [
    {"sha": c["sha"][:7], "date": c["commit"]["committer"]["date"],
     "msg": c["commit"]["message"].splitlines()[0][:120]}
    for c in commits
]
# 2. synced/h24 listing
listing = gh_api.get_file(OWNER, REPO, "synced/h24", ref=BRANCH)
out["h24_files"] = [e["name"] for e in listing] if isinstance(listing, list) else listing
# 3. bigeval_live listing
try:
    live = gh_api.get_file(OWNER, REPO, "synced/h24/bigeval_live", ref=BRANCH)
    out["live_files"] = [e["name"] for e in live] if isinstance(live, list) else live
except Exception as e:
    out["live_files_error"] = str(e)

# 4. read the m3s2r1_e0 live file (may be large; use raw blob via git blob API if needed)
def read_file_text(path):
    f = gh_api.get_file(OWNER, REPO, path, ref=BRANCH)
    if isinstance(f, dict) and "content" in f:
        return base64.b64decode(f["content"]).decode("utf-8", errors="replace")
    return None

def live_stats(path):
    txt = read_file_text(path)
    if txt is None:
        return {"error": "unreadable"}
    lines = [l for l in txt.splitlines() if l.strip()]
    wins = 0
    margins = []
    last = None
    for l in lines:
        try:
            j = json.loads(l)
        except Exception:
            continue
        last = j
        if isinstance(j, dict):
            # count wins by margin>0 where present
            m = j.get("margin", j.get("score_diff"))
            if isinstance(m, (int, float)):
                margins.append(m)
                if m > 0:
                    wins += 1
    import statistics
    return {"lines": len(lines), "wins": wins,
            "mean_margin": round(statistics.mean(margins), 1) if margins else None,
            "last": last}

for name in ["m3s2r1_e0.jsonl", "echo2_smoke.jsonl"]:
    p = f"synced/h24/bigeval_live/{name}"
    try:
        out[name] = live_stats(p)
    except Exception as e:
        out[name] = {"error": str(e)}

# 5. cal.out files full text
for name in ["m3s2r1_cal.out", "m3s2r1_e0_cal.out"]:
    p = f"synced/h24/{name}"
    try:
        txt = read_file_text(p)
        out[name] = txt if txt is not None else "MISSING"
    except Exception as e:
        out[name] = f"ERROR {e}"

print(json.dumps(out, indent=1, ensure_ascii=False))
