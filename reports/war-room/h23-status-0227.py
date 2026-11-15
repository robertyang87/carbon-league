"""H23 status check driver (02:27 run): read sync/h29 + sync/h23 via GitHub API
(5 calls: rate_limit, 2 branch heads, 2 dir listings), fetch the 8 live files
via raw.githubusercontent (no API quota), count lines + unique seeds + last line.
All network ops bounded with timeouts.
"""
import sys, json, urllib.request, urllib.error
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
HF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
LOG = f"{HF}/h23-status-0227.log"

def api(path):
    return gh_api.req("GET", path)

out = []
def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)

rl = api("/rate_limit")
core = rl["resources"]["core"]
p(f"RATE core remaining={core['remaining']}/{core['limit']} reset={core['reset']}")

heads = {}
for br in ("sync/h29", "sync/h23"):
    b = api(f"/repos/{OWNER}/{REPO}/branches/{br}")
    c = b["commit"]["commit"]
    heads[br] = (b["commit"]["sha"][:8], c["committer"]["date"])
    p(f"BRANCH {br} sha={b['commit']['sha'][:8]} date={c['committer']['date']} msg={c['message'][:100]!r}")

live_dir = api(f"/repos/{OWNER}/{REPO}/contents/synced/h29/bigeval_live?ref=sync/h29")
p(f"LIVE-DIR entries={len(live_dir)}")
for e in live_dir:
    p(f"  {e['name']} size={e['size']} type={e['type']}")

root_dir = api(f"/repos/{OWNER}/{REPO}/contents/synced/h29?ref=sync/h29")
names = [e["name"] for e in root_dir]
p(f"H29-ROOT entries={len(root_dir)}: {names}")
cal = [n for n in names if n.endswith("_cal.out") or n.endswith(".cal.out")]
p(f"CAL-OUT files: {cal if cal else 'none'}")

summ = api(f"/repos/{OWNER}/{REPO}/contents/synced/h29/summaries?ref=sync/h29")
p(f"SUMMARIES entries={len(summ)}: {[e['name'] for e in summ]}")

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
        p(f"  {e['name']}: RAW FETCH FAILED {ex!r}")
        continue
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    seeds = []
    last_seed = last_ts = None
    for ln in lines:
        try:
            seeds.append(json.loads(ln).get("seed"))
        except Exception:
            pass
    if lines:
        try:
            last = json.loads(lines[-1])
            last_seed = last.get("seed")
            last_ts = last.get("ts") or last.get("time") or last.get("timestamp") or last.get("ended_at")
        except Exception:
            last_seed = lines[-1][:100]
    p(f"  {e['name']}: lines={len(lines)} unique_seeds={len(set(seeds))} last_seed={last_seed} last_ts={last_ts}")
    total += len(lines)
p(f"TOTAL live lines = {total} / 1600")

open(LOG, "w").write("\n".join(out) + "\n")
p("saved", LOG)
