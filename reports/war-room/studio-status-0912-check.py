import sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
# Read-only: list recent cloud-assistant.yml runs, no dispatch, no claim polling.
# War-room order stands: no dispatch/claim retries while channel down.
# Trip marker: STUDIOSTAT0912 (read-only check only, no probe dispatched).
try:
    r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=10")
except Exception as e:
    print("LIST-ERROR:", repr(e))
    raise SystemExit(1)
for run in r.get("workflow_runs", []):
    print(run["id"], run.get("created_at"), run.get("status"), run.get("conclusion"), run.get("display_title", "")[:60])
print("total_count:", r.get("total_count"))
