"""H23 status 10:12 direct read (run 2026-11-14 10:12 CST).
sync/h23 head + synced/h23 listing + bigeval_live count + 3 cal.out contents
(expected silence: spot-23 released 11-12 20:25-21:00).
main head; sync/h29 head + 6 commits + live files (8 legacy + k0mon 2);
sync/h31 head. Marker: H23STATUS1012. ~22 core API calls (rate_limit free).
"""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
HF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
LOG = f"{HF}/h23-status-1012-1114.log"

def api(path):
    return gh_api.req("GET", path)

out = []
def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)

p("MARKER H23STATUS1012")

def branch_head(br):
    try:
        b = api(f"/repos/{OWNER}/{REPO}/branches/{br}")
        c = b["commit"]
        p(f"BRANCH_HEAD {br} {c['sha'][:8]} {c['commit']['committer']['date']} {c['commit']['message'].splitlines()[0][:90]!r}")
        return c["sha"][:8]
    except Exception as e:
        p(f"BRANCH_HEAD {br} ERROR {type(e).__name__}: {e}")
        return None

branch_head("sync/h23")
branch_head("main")
branch_head("sync/h29")
branch_head("sync/h31")

try:
    commits = api(f"/repos/{OWNER}/{REPO}/commits?sha=sync/h29&per_page=6")
    p(f"COMMITS sync/h29 (newest first, {len(commits)}):")
    for c in commits:
        p(f"  {c['sha'][:8]} {c['commit']['committer']['date']} {c['commit']['message'].splitlines()[0][:90]!r}")
except Exception as e:
    p(f"COMMITS sync/h29 ERROR {type(e).__name__}: {e}")

try:
    d = api(f"/repos/{OWNER}/{REPO}/contents/synced/h23?ref=sync/h23")
    names = sorted(x["name"] for x in d) if isinstance(d, list) else [str(d)]
    p(f"DIR synced/h23: count={len(names)}")
except Exception as e:
    p(f"DIR synced/h23 ERROR {type(e).__name__}: {e}")

try:
    d = api(f"/repos/{OWNER}/{REPO}/contents/synced/h23/bigeval_live?ref=sync/h23")
    names = sorted(x["name"] for x in d) if isinstance(d, list) else [str(d)]
    p(f"DIR synced/h23/bigeval_live: count={len(names)} tail={names[-5:]}")
except Exception as e:
    p(f"DIR synced/h23/bigeval_live ERROR {type(e).__name__}: {e}")

for cal in ["lineA_cal.out", "m3s2r1pre_cal.out", "m7bsr1_cal.out"]:
    try:
        f = api(f"/repos/{OWNER}/{REPO}/contents/synced/h23/{cal}?ref=sync/h23")
        raw = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
        p(f"CAL {cal}: size={f.get('size')} content={raw.strip()[:400]!r}")
    except Exception as e:
        p(f"CAL {cal} ERROR {type(e).__name__}: {e}")

try:
    d = api(f"/repos/{OWNER}/{REPO}/contents/synced/h29?ref=sync/h29")
    names = sorted(x["name"] for x in d) if isinstance(d, list) else [str(d)]
    p("DIR synced/h29:", ", ".join(names))
except Exception as e:
    p(f"DIR synced/h29 ERROR {type(e).__name__}: {e}")

files = ["fableg1_echo2.jsonl","fableg1_m2.jsonl","fabler1_echo2.jsonl","fabler1_m2.jsonl",
         "m7bsv2g1_echo2.jsonl","m7bsv2g1_m2.jsonl","m7bsv2r1_echo2.jsonl","m7bsv2r1_m2.jsonl",
         "k0_a0_b0.jsonl","k0_a1_b0.jsonl"]
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
p(f"TOTAL live lines = {total} / 1728")
try:
    rl = api("/rate_limit")
    p("RATE remaining:", rl["resources"]["core"]["remaining"], "reset:", rl["resources"]["core"]["reset"])
except Exception as e:
    p(f"RATE ERROR {type(e).__name__}: {e}")
open(LOG, "w").write("\n".join(out) + "\n")
p("saved", LOG)
