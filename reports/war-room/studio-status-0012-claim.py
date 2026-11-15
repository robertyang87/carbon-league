import sys
sys.path.insert(0, "/home/hatch/workspace")
sys.path.insert(0, "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files")
import gh_api, time
from fetch_job_log_ranges import fetch_log
MARK = "STUDIOSTAT0012"
CUTOFF = "2026-11-12T16:16:00Z"  # dispatch 00:16:07 CST
r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
cands = [run for run in r.get("workflow_runs", []) if run["created_at"] >= CUTOFF]
print("runs since dispatch:", len(cands), flush=True)
found = None
for run in cands:
    print(run["id"], run["created_at"], run["status"], run.get("conclusion"), flush=True)
    if run["status"] != "completed":
        continue
    try:
        jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{run['id']}/jobs")
    except Exception as e:
        print("jobs error:", e, flush=True); continue
    for j in jobs.get("jobs", []):
        try:
            log = fetch_log(j["id"])
        except Exception as e:
            print(run["id"], j["id"], "range fetch error:", e, flush=True); continue
        print(run["id"], j["id"], "bytes", len(log), "marker" if MARK in log else "no-marker", flush=True)
        if MARK in log:
            i0 = log.find(MARK)
            print(log[max(0, i0-200):i0+16000], flush=True)
            found = (run["id"], j["id"])
            break
    if found:
        break
print("FOUND" if found else "NOT-FOUND", found, flush=True)
