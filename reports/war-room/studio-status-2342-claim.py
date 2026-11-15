import sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api, time
MARK = "STUDIOSTAT2342IT320"
r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
for run in r.get("workflow_runs", [])[:12]:
    print(run["id"], run["status"], run.get("conclusion"), run.get("created_at"), run.get("name"))
