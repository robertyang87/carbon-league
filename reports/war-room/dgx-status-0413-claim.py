"""DGX status 0413 targeted claim: list cloud-assistant runs created after
dispatch (2026-11-12T20:14:05Z), print id/created_at/status/conclusion,
then fetch jobs+logs for the run created closest after dispatch.
Single pass, no polling loop (quota-throttle note 2026-11-13 04:14)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/workspace"))
import gh_api

MARK = "DGXSTATUS0413A"
r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
runs = r.get("workflow_runs", [])
print(f"total listed: {len(runs)}")
cands = [x for x in runs if x["created_at"] >= "2026-11-12T20:12:00Z"]
for x in sorted(cands, key=lambda y: y["created_at"]):
    print(x["id"], x["created_at"], x["status"], x.get("conclusion"), x.get("display_title", "")[:60])
# candidate: first run created at/after dispatch time
mine = [x for x in cands if x["created_at"] >= "2026-11-12T20:14:05Z"]
mine = sorted(mine, key=lambda y: y["created_at"])
if not mine:
    print("NO-RUN-CREATED-AFTER-DISPATCH")
    sys.exit(0)
target = mine[0]
print("TARGET", target["id"], target["created_at"], target["status"], target.get("conclusion"))
jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{target['id']}/jobs")
for j in jobs.get("jobs", []):
    print("JOB", j["id"], j.get("name"), j.get("status"), j.get("conclusion"), j.get("started_at"))
    try:
        log = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
        print(f"  log bytes={len(log)} marker={'YES' if MARK in log else 'NO'}")
        if MARK in log:
            i0 = log.find(MARK)
            print(log[max(0, i0 - 200):i0 + 12000])
            with open(os.path.expanduser("~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-status-0413.log"), "w") as f:
                f.write(log)
            print("CLAIMED")
    except Exception as e:
        print("  log error:", e)
