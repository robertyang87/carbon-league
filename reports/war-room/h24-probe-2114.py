"""H24 probe 21:14 (marker H24PROBE2114): direct-read sync/h24 branch state."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import req, get_file

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h24"
print("MARKER=H24PROBE2114")

# 1. latest commit on sync/h24 branch
commits = req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={BR}&per_page=1")
c = commits[0]
print("LATEST_COMMIT:", c["sha"][:7], c["commit"]["author"]["date"], "|", c["commit"]["message"].splitlines()[0])

# 2. list synced/h24/ and synced/h24/bigeval_live/
for d in ["synced/h24", "synced/h24/bigeval_live"]:
    try:
        listing = get_file(OWNER, REPO, d, ref=BR)
        print(f"DIR {d}:")
        for item in listing:
            print("  ", item["type"], item["name"], item.get("size"))
    except Exception as e:
        print(f"DIR {d} ERROR:", e)

# 3. m3s2r1_cal.out full text (if present)
def fetch_text(path):
    f = get_file(OWNER, REPO, path, ref=BR)
    if isinstance(f, dict) and f.get("content"):
        return base64.b64decode(f["content"]).decode("utf-8", errors="replace")
    return None

for p in ["synced/h24/m3s2r1_cal.out", "synced/h24/bigeval_live/k4v2r1_e0.jsonl",
          "synced/h24/bigeval_live/m3s2r1_echo2.jsonl",
          "synced/h24/k4v2_chain.log", "synced/h24/k4v2r1_e0.launcher.log"]:
    try:
        t = fetch_text(p)
        if t is None:
            print(f"FILE {p}: <not found / no content>")
        else:
            lines = t.splitlines()
            print(f"FILE {p}: {len(lines)} lines, {len(t)} bytes")
            if "cal.out" in p:
                print("--- cal.out full text ---")
                print(t)
                print("--- end ---")
            elif lines:
                print("  first:", lines[0][:200])
                print("  last :", lines[-1][:200])
    except Exception as e:
        print(f"FILE {p} ERROR:", repr(e))
