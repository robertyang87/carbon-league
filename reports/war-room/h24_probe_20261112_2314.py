"""H24 targeted probe 2026-11-12 23:14 run. Marker: H24PROBE2314."""
import sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"

def commits_of(branch, n=3):
    try:
        cs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={branch}&per_page={n}")
        for c in cs:
            print("COMMIT", branch, c["sha"][:7], c["commit"]["committer"]["date"],
                  (c["commit"]["message"].splitlines()[0] if c["commit"]["message"] else ""))
    except Exception as e:
        print("COMMIT", branch, "ERR:", e)

commits_of("sync/h24")
commits_of("sync/h28")

def get_raw(path, branch):
    try:
        f = gh_api.get_file(OWNER, REPO, path, ref=branch)
        if isinstance(f, dict) and f.get("content"):
            return base64.b64decode(f["content"]).decode("utf-8", errors="replace")
        return None
    except Exception as e:
        return f"ERR: {e}"

# sync/h28 live files for K4 probe
for p in ["synced/h28/bigeval_live/k4v2r1_e0.jsonl",
          "synced/h28/bigeval_live/k4v2g1_e0.jsonl"]:
    raw = get_raw(p, "sync/h28")
    if raw is None:
        print("LIVE", p, "MISSING/EMPTY")
    elif raw.startswith("ERR"):
        print("LIVE", p, raw)
    else:
        lines = [l for l in raw.splitlines() if l.strip()]
        print(f"LIVE {p} lines={len(lines)}")
        if lines:
            print("  first:", lines[0][:300])
            print("  last:", lines[-1][:300])

# sync/h28 tree listing (full, to catch cal.out / logs)
try:
    tree = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/git/trees/sync/h28?recursive=1")
    paths = [t["path"] for t in tree.get("tree", [])]
    print("H28TREE count=", len(paths))
    for p in paths:
        print("H28", p)
except Exception as e:
    print("H28TREE ERR:", e)

# any cal.out on sync/h28
for p in ["synced/h28/k4v2r1_cal.out", "synced/h28/k4v2r1_e0_cal.out"]:
    raw = get_raw(p, "sync/h28")
    if raw is None:
        print("FILE", p, "MISSING/EMPTY")
    elif raw.startswith("ERR"):
        print("FILE", p, raw)
    else:
        print(f"FILE {p} bytes={len(raw)}")
        print(raw[:3000])
        print("---end---")

# sync/h24 unchanged check: last exam cal.out size
raw = get_raw("synced/h24/m3s2r1_cal.out", "sync/h24")
if raw is None:
    print("FILE synced/h24/m3s2r1_cal.out MISSING/EMPTY")
elif raw.startswith("ERR"):
    print("FILE synced/h24/m3s2r1_cal.out", raw)
else:
    print(f"FILE synced/h24/m3s2r1_cal.out bytes={len(raw)}")
