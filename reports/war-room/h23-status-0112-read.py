#!/usr/bin/env python3
"""H23 status trip 01:12 - direct API read of sync/h29 backfill files.
Reads branch head, bigeval_live dir listing, and the 8 per-game line files.
Counts rows and checks seed domains (m2: 996000-996199, echo2: 999000-999199).
Prints a compact JSON summary. ~11 core API calls."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h29"
FILES = [
    "fabler1_m2", "fabler1_echo2",
    "fableg1_m2", "fableg1_echo2",
    "m7bsv2r1_m2", "m7bsv2r1_echo2",
    "m7bsv2g1_m2", "m7bsv2g1_echo2",
]

out = {"api_calls": 0}
try:
    rl = gh_api.req("GET", "/rate_limit")
    out["rate_limit_core"] = rl["resources"]["core"]
except Exception as e:
    out["rate_limit_error"] = str(e)

try:
    b = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{BRANCH.replace('/', '%2F')}")
    out["api_calls"] += 1
    out["branch_head"] = b["commit"]["sha"][:8]
    out["branch_head_time"] = b["commit"]["commit"]["author"]["date"]
    out["branch_head_msg"] = b["commit"]["commit"]["message"].splitlines()[0][:120]
except Exception as e:
    out["branch_error"] = str(e)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    sys.exit(0)

# list bigeval_live dir
try:
    listing = gh_api.get_file(OWNER, REPO, "synced/h29/bigeval_live", ref=BRANCH)
    out["api_calls"] += 1
    out["dir"] = [{"name": f["name"], "size": f["size"]} for f in listing if f["type"] == "file"]
except Exception as e:
    out["dir_error"] = str(e)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    sys.exit(0)

out["files"] = {}
total_true = 0
for tag in FILES:
    path = f"synced/h29/bigeval_live/{tag}.jsonl"
    try:
        f = gh_api.get_file(OWNER, REPO, path, ref=BRANCH)
        out["api_calls"] += 1
        raw = base64.b64decode(f["content"]).decode("utf-8", "replace")
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        seeds = []
        bad = []
        for ln in lines:
            try:
                d = json.loads(ln)
                s = d.get("seed")
                seeds.append(s)
                lo = 996000 if tag.endswith("_m2") else 999000
                if not (isinstance(s, int) and lo <= s <= lo + 199):
                    bad.append(s)
            except Exception:
                bad.append("unparsable")
        out["files"][tag] = {
            "rows": len(lines),
            "sha": f["sha"][:8],
            "seeds_min": min(seeds) if seeds else None,
            "seeds_max": max(seeds) if seeds else None,
            "out_of_domain": bad,
            "last_seed": seeds[-1] if seeds else None,
        }
        total_true += len(lines) - len(bad)
    except Exception as e:
        out["files"][tag] = {"error": str(e)[:200]}
out["total_rows_raw"] = sum(v.get("rows", 0) for v in out["files"].values())
out["total_rows_true"] = total_true

# summaries dir listing (finals check, no content fetch - known stable)
try:
    sl = gh_api.get_file(OWNER, REPO, "synced/h29/summaries", ref=BRANCH)
    out["api_calls"] += 1
    out["summaries"] = [f["name"] for f in sl if f["type"] == "file"]
except Exception as e:
    out["summaries_error"] = str(e)

# any *_cal.out under synced/h29 root?
try:
    root = gh_api.get_file(OWNER, REPO, "synced/h29", ref=BRANCH)
    out["api_calls"] += 1
    out["root_entries"] = [f["name"] for f in root]
    out["cal_out"] = [f["name"] for f in root if f["name"].endswith("_cal.out")]
except Exception as e:
    out["root_error"] = str(e)

print(json.dumps(out, ensure_ascii=False, indent=1))
