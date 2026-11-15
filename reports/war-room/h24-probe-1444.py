"""h24-status-report probe, trip h24-1444, 2026-11-14 ~14:44 CST.
Direct read of robertyang87/carbon-league sync/h24 via gh_api.
Checks: branch head, synced/h24 top-level count, bigeval_live count,
*_cal.out files (full text), echo2 live file recompute, rate limit.
"""
import base64, json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h24"
out = {}

# 1. branch head
br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{BR}")
out["branch_head_sha"] = br["commit"]["sha"]
out["branch_head_date"] = br["commit"]["commit"]["committer"]["date"]

# 2. synced/h24 top-level listing
top = gh_api.get_file(OWNER, REPO, "synced/h24", ref=BR)
out["top_count"] = len(top) if isinstance(top, list) else -1
out["top_names"] = sorted(e["name"] for e in top) if isinstance(top, list) else str(top)[:200]

# 3. bigeval_live listing
live = gh_api.get_file(OWNER, REPO, "synced/h24/bigeval_live", ref=BR)
out["live_count"] = len(live) if isinstance(live, list) else -1

# 4. cal.out files at top level
cal_files = [e["name"] for e in top if isinstance(top, list) and e["name"].endswith("_cal.out")]
out["cal_files"] = cal_files
for name in cal_files:
    f = gh_api.get_file(OWNER, REPO, f"synced/h24/{name}", ref=BR)
    raw = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
    out[f"cal_{name}"] = {"size": f["size"], "sha": f["sha"], "text": raw}

# 5. echo2 live file recompute (single fetch)
f = gh_api.get_file(OWNER, REPO, "synced/h24/bigeval_live/m3s2r1_echo2.jsonl", ref=BR)
raw = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
lines = [ln for ln in raw.splitlines() if ln.strip()]
wins = 0; margins = []; last = None; errs = 0
for ln in lines:
    try:
        d = json.loads(ln)
    except Exception:
        errs += 1; continue
    m = d.get("margin")
    if m is not None:
        margins.append(m)
        if m > 0:
            wins += 1
    last = d
out["echo2"] = {
    "size": f["size"], "sha": f["sha"], "n": len(lines),
    "wins": wins,
    "mean": round(sum(margins)/len(margins), 1) if margins else None,
    "last_seed": last.get("seed") if last else None,
    "last_margin": last.get("margin") if last else None,
    "last_time": last.get("time") or last.get("ts") or last.get("timestamp") if last else None,
    "errs": errs,
}

# 6. rate limit
rl = gh_api.req("GET", "/rate_limit")
out["rate_core_remaining"] = rl["resources"]["core"]["remaining"]
out["rate_core_limit"] = rl["resources"]["core"]["limit"]

print(json.dumps(out, ensure_ascii=False, indent=1))
