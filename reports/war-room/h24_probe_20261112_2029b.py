"""H24 targeted probe 2026-11-12 20:29 run (part 2). Marker: H24PROBE2029B."""
import json, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h24"

commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=3")
for c in commits:
    print("COMMIT", c["sha"][:7], c["commit"]["committer"]["date"],
          (c["commit"]["message"].splitlines()[0] if c["commit"]["message"] else ""))

def get_raw(path):
    try:
        f = gh_api.get_file(OWNER, REPO, path, ref=BRANCH)
        if isinstance(f, dict) and f.get("content"):
            return base64.b64decode(f["content"]).decode("utf-8", errors="replace")
        return None
    except Exception as e:
        return f"ERR: {e}"

for p in ["synced/h24/bigeval_live/k4v2r1_e0.jsonl",
          "synced/h24/bigeval_live/m3s2r1_echo2.jsonl",
          "synced/h24/bigeval_live/m3s2r1_m2.jsonl",
          "synced/h24/bigeval_live/m3s2r1_e0.jsonl"]:
    raw = get_raw(p)
    if raw is None:
        print("LIVE", p, "MISSING/EMPTY")
    elif raw.startswith("ERR"):
        print("LIVE", p, raw)
    else:
        lines = [l for l in raw.splitlines() if l.strip()]
        print(f"LIVE {p} lines={len(lines)}")
        if lines:
            print("  first:", lines[0][:200])
            print("  last:", lines[-1][:200])

for p in ["synced/h24/m3s2r1_cal.out",
          "synced/h24/k4v2_chain.log",
          "synced/h24/k4v2r1_e0.launcher.log",
          "synced/h24/m3s2r1_echo2.launcher.log"]:
    raw = get_raw(p)
    if raw is None:
        print("FILE", p, "MISSING/EMPTY")
    elif raw.startswith("ERR"):
        print("FILE", p, raw)
    else:
        print(f"FILE {p} bytes={len(raw)}")
        print(raw[:3000])
        print("---end---")
