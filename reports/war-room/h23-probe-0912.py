"""H23 status probe 09:12 - branch heads + latest Actions runs + rate limit. Marker H23STAT0912."""
import sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
print("MARKER: H23STAT0912")

for br in ["sync/h29", "sync/h23"]:
    try:
        commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={br}&per_page=1")
        c = commits[0]
        print(f"HEAD {br}: {c['sha'][:7]} {c['commit']['committer']['date']} {c['commit']['message'][:100]}")
    except Exception as e:
        print(f"HEAD {br}: ERROR {e}")

try:
    runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs?per_page=5")
    print(f"RUNS total_count={runs.get('total_count')}")
    for r in runs.get("workflow_runs", []):
        print(f"  run {r['id']} {r['name']} {r['created_at']} status={r['status']} conclusion={r['conclusion']}")
except Exception as e:
    print(f"RUNS: ERROR {e}")

try:
    rl = gh_api.req("GET", "/rate_limit")
    core = rl["resources"]["core"]
    print(f"RATE core remaining={core['remaining']}/5000 reset={core['reset']}")
except Exception as e:
    print(f"RATE: ERROR {e}")
