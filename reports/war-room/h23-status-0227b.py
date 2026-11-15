"""H23 status 02:27 fallback: contents API (base64) read of the 8 live files
(raw.githubusercontent 404s persistently - same as 02:12). Plus recent commits
on sync/h29 to characterize the 02:00:21 -> 02:27:58 gap. ~10 API calls.
"""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
HF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
LOG = f"{HF}/h23-status-0227b.log"

def api(path):
    return gh_api.req("GET", path)

out = []
def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)

# recent commits on sync/h29
commits = api(f"/repos/{OWNER}/{REPO}/commits?sha=sync/h29&per_page=12")
p(f"COMMITS sync/h29 (newest first, {len(commits)}):")
for c in commits:
    p(f"  {c['sha'][:8]} {c['commit']['committer']['date']} {c['commit']['message'].splitlines()[0][:90]!r}")

files = ["fableg1_echo2.jsonl","fableg1_m2.jsonl","fabler1_echo2.jsonl","fabler1_m2.jsonl",
         "m7bsv2g1_echo2.jsonl","m7bsv2g1_m2.jsonl","m7bsv2r1_echo2.jsonl","m7bsv2r1_m2.jsonl"]
total = 0
for name in files:
    f = api(f"/repos/{OWNER}/{REPO}/contents/synced/h29/bigeval_live/{name}?ref=sync/h29")
    raw = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
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
    p(f"  {name}: lines={len(lines)} unique_seeds={len(set(seeds))} last_seed={last_seed} last_ts={last_ts}")
    total += len(lines)
p(f"TOTAL live lines = {total} / 1600")
rl = api("/rate_limit")
p("RATE remaining:", rl["resources"]["core"]["remaining"])
open(LOG, "w").write("\n".join(out) + "\n")
p("saved", LOG)
