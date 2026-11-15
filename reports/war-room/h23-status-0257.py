"""H23 status 02:42 direct read: sync/h29 head + recent commits + 8 live files
via contents API (raw.githubusercontent 404s persistently). Plus sync/h23 head
(expected silence) and synced/h29 dir listing for *_cal.out / summaries.
Marker: H23STATUS0257. ~16 API calls.
"""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
HF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
LOG = f"{HF}/h23-status-0257.log"

def api(path):
    return gh_api.req("GET", path)

out = []
def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)

p("MARKER H23STATUS0257")
try:
    br = api(f"/repos/{OWNER}/{REPO}/branches/sync/h29")
    c = br["commit"]
    p(f"BRANCH_HEAD sync/h29 {c['sha'][:8]} {c['commit']['committer']['date']} {c['commit']['message'].splitlines()[0][:90]!r}")
except Exception as e:
    p(f"BRANCH_HEAD sync/h29 ERROR {type(e).__name__}: {e}")

try:
    br = api(f"/repos/{OWNER}/{REPO}/branches/sync/h23")
    c = br["commit"]
    p(f"BRANCH_HEAD sync/h23 {c['sha'][:8]} {c['commit']['committer']['date']} {c['commit']['message'].splitlines()[0][:90]!r}")
except Exception as e:
    p(f"BRANCH_HEAD sync/h23 ERROR {type(e).__name__}: {e}")

try:
    commits = api(f"/repos/{OWNER}/{REPO}/commits?sha=sync/h29&per_page=6")
    p(f"COMMITS sync/h29 (newest first, {len(commits)}):")
    for c in commits:
        p(f"  {c['sha'][:8]} {c['commit']['committer']['date']} {c['commit']['message'].splitlines()[0][:90]!r}")
except Exception as e:
    p(f"COMMITS ERROR {type(e).__name__}: {e}")

try:
    d = api(f"/repos/{OWNER}/{REPO}/contents/synced/h29?ref=sync/h29")
    names = sorted(x["name"] for x in d) if isinstance(d, list) else [str(d)]
    p("DIR synced/h29:", ", ".join(names))
    cals = [n for n in names if n.endswith("_cal.out") or n.endswith(".out")]
    p("CAL_OUT files:", cals if cals else "none")
except Exception as e:
    p(f"DIR ERROR {type(e).__name__}: {e}")

files = ["fableg1_echo2.jsonl","fableg1_m2.jsonl","fabler1_echo2.jsonl","fabler1_m2.jsonl",
         "m7bsv2g1_echo2.jsonl","m7bsv2g1_m2.jsonl","m7bsv2r1_echo2.jsonl","m7bsv2r1_m2.jsonl"]
total = 0
for name in files:
    try:
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
    except Exception as e:
        p(f"  {name}: ERROR {type(e).__name__}: {e}")
p(f"TOTAL live lines = {total} / 1600")
try:
    rl = api("/rate_limit")
    p("RATE remaining:", rl["resources"]["core"]["remaining"])
except Exception as e:
    p(f"RATE ERROR {type(e).__name__}: {e}")
open(LOG, "w").write("\n".join(out) + "\n")
p("saved", LOG)
