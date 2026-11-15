"""H23 status check driver (02:12 run): read sync/h29 + sync/h23 via GitHub API
(4 calls: rate_limit, 2 branch heads, 2 dir listings), fetch the 8 live files
via raw.githubusercontent (no API quota), count lines + unique seeds + last line.
"""
import sys, json, urllib.request, urllib.error
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"

def api(path):
    return gh_api.req("GET", path)

rl = api("/rate_limit")
core = rl["resources"]["core"]
print(f"RATE core remaining={core['remaining']}/{core['limit']} reset={core['reset']}")

for br in ("sync/h29", "sync/h23"):
    b = api(f"/repos/{OWNER}/{REPO}/branches/{br}")
    c = b["commit"]["commit"]
    print(f"BRANCH {br} sha={b['commit']['sha'][:8]} date={c['committer']['date']} msg={c['message'][:80]!r}")

live_dir = api(f"/repos/{OWNER}/{REPO}/contents/synced/h29/bigeval_live?ref=sync/h29")
print(f"LIVE-DIR entries={len(live_dir)}")
for e in live_dir:
    print(f"  {e['name']} size={e['size']} type={e['type']}")

root_dir = api(f"/repos/{OWNER}/{REPO}/contents/synced/h29?ref=sync/h29")
names = [e["name"] for e in root_dir]
print(f"H23-DIR entries={len(root_dir)}: {names}")
cal = [n for n in names if n.endswith("_cal.out") or n.endswith(".cal.out")]
print(f"CAL-OUT files: {cal if cal else 'none'}")

RAW = "https://raw.githubusercontent.com/robertyang87/carbon-league/sync/h29/synced/h29/bigeval_live/"
total = 0
for e in live_dir:
    if e["type"] != "file":
        continue
    url = RAW + e["name"]
    req = urllib.request.Request(url, headers={"User-Agent": "muse-agent"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as ex:
        print(f"  {e['name']}: RAW FETCH FAILED {ex!r}")
        continue
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    seeds = []
    last_seed = last_ts = None
    for ln in lines:
        try:
            obj = json.loads(ln)
            s = obj.get("seed")
            seeds.append(s)
        except Exception:
            pass
    if lines:
        try:
            last = json.loads(lines[-1])
            last_seed = last.get("seed")
            last_ts = last.get("ts") or last.get("time") or last.get("timestamp")
        except Exception:
            last_seed = lines[-1][:120]
    print(f"  {e['name']}: lines={len(lines)} unique_seeds={len(set(seeds))} last_seed={last_seed} last_ts={last_ts}")
    total += len(lines)
print(f"TOTAL live lines = {total} / 1600")
