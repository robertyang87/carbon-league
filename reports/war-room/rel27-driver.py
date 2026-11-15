#!/usr/bin/env python3
"""Dispatch ecs-release.yml for spot-27 and claim the result ONLY by marker in logs."""
import sys, time, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, WF = "robertyang87", "carbon-league", "ecs-release.yml"
MARKER = "REL27M3Q8Z52"
INPUTS = {
    "instance_id": "i-uf6bl5muucd5dj2bh200",
    "region": "cn-shanghai",
    "expected_ip": "47.117.49.49",
    "marker": MARKER,
}

t0 = time.time() - 60  # created_at cutoff (allow clock skew)
gh_api.dispatch(OWNER, REPO, WF, ref="main", inputs=INPUTS)
print("DISPATCHED marker=", MARKER, flush=True)

KEYS = (MARKER, "PRECHECK", "DELETE_", "VERIFY_", "FOUND", "RESULT=")

deadline = time.time() + 540
claimed = None
while time.time() < deadline:
    time.sleep(20)
    runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30")
    cand = []
    for run in runs.get("workflow_runs", []):
        # created_at like 2026-11-12T12:52:00Z
        ca = run.get("created_at", "")
        try:
            ts = time.mktime(time.strptime(ca, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
        except Exception:
            continue
        if ts >= t0:
            cand.append(run)
    for run in cand:
        if run.get("status") != "completed":
            continue
        jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{run['id']}/jobs")
        for job in jobs.get("jobs", []):
            try:
                logs = gh_api.get_job_logs(OWNER, REPO, job["id"])
            except Exception as e:
                print("LOGFETCH_ERR", run["id"], e, flush=True)
                continue
            if MARKER in logs:
                claimed = (run["id"], job["id"], run.get("conclusion"), logs)
                break
        if claimed:
            break
    if claimed:
        break
    print("...waiting, candidate runs:", [(r["id"], r["status"]) for r in cand], flush=True)

if not claimed:
    print("NO_CLAIM: no completed run containing marker before deadline")
    sys.exit(1)

run_id, job_id, conclusion, logs = claimed
print(f"RUN_ID={run_id} JOB_ID={job_id} CONCLUSION={conclusion}")
for line in logs.splitlines():
    if any(k in line for k in KEYS):
        # strip the leading timestamp GitHub adds to log lines
        print(line.split(" ", 1)[-1] if "T" in line[:30] else line)
