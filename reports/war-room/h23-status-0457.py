"""H23 status 04:42 lean direct read (quota discipline: core was 3072 @04:27).
- sync/h29 + sync/h23 branch heads (2 calls).
- If h29 head unchanged from 53f3c64f @04:12:54, live files are byte-identical
  (contents versioned by commit) -> skip the 8 file reads, total stays 664/1600.
- Recent Actions runs (1 call) to test whether the runner-allocation outage
  (down since 04:13) has recovered.
- rate_limit is free.
Marker: H23STATUS0457.
"""
import sys, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
HF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
LOG = f"{HF}/h23-status-0457.log"

out = []
def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)

p("MARKER H23STATUS0457")
try:
    br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/sync/h29")
    c = br["commit"]
    p(f"BRANCH_HEAD sync/h29 {c['sha'][:8]} {c['commit']['committer']['date']} {c['commit']['message'].splitlines()[0][:90]!r}")
except Exception as e:
    p(f"BRANCH_HEAD sync/h29 ERROR {type(e).__name__}: {e}")

try:
    br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/sync/h23")
    c = br["commit"]
    p(f"BRANCH_HEAD sync/h23 {c['sha'][:8]} {c['commit']['committer']['date']} {c['commit']['message'].splitlines()[0][:90]!r}")
except Exception as e:
    p(f"BRANCH_HEAD sync/h23 ERROR {type(e).__name__}: {e}")

try:
    runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs?per_page=5")
    for r in runs.get("workflow_runs", [])[:5]:
        p(f"RUN id={r['id']} name={r['name'][:40]} status={r['status']} conclusion={r['conclusion']} created={r['created_at']} event={r['event']}")
except Exception as e:
    p(f"RUNS ERROR {type(e).__name__}: {e}")

try:
    rl = gh_api.req("GET", "/rate_limit")
    p("RATE remaining:", rl["resources"]["core"]["remaining"],
      "reset:", rl["resources"]["core"]["reset"])
except Exception as e:
    p(f"RATE ERROR {type(e).__name__}: {e}")
open(LOG, "w").write("\n".join(out) + "\n")
p("saved", LOG)
