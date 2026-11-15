"""Studio 04:12 status probe — retry dispatch + claim (robust claim loop).

Dispatches studio-status-0412.sh via cloud-assistant.yml, then polls the
run until completed and fetches the job log once (avoids the old driver's
crash when fetching logs of a not-yet-started job). Verdict from the
STUDIO0412 marker + STUDIO-END canary.
Usage: studio-status-0412-driver.py
"""
import sys, os, base64, time
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
from datetime import datetime, timezone

HF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"
INSTANCE, REGION = "i-uf62k0o6zbe6j7zxnzg5", "cn-shanghai"
SCRIPT = os.path.join(HF, "studio-status-0412.sh")
MARK = "STUDIO0412"

script = open(SCRIPT).read()
print(f"script bytes: {len(script)} b64: {len(base64.b64encode(script.encode()))} marker: {MARK}", flush=True)
rl = gh_api.req("GET", "/rate_limit")
print("rate_limit core remaining:", rl["resources"]["core"]["remaining"], flush=True)

t0 = datetime.now(timezone.utc)
resp = gh_api.dispatch(OWNER, REPO, WF, ref="main",
    inputs={"instance_id": INSTANCE, "region": REGION,
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched at", t0.strftime("%H:%M:%S"), "UTC resp:", resp, flush=True)

# Find our run: newest workflow_dispatch run created after t0.
run_id = None
for _ in range(12):
    runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=5")
    for r in runs.get("workflow_runs", []):
        created = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
        if created >= t0:
            run_id = r["id"]
            break
    if run_id:
        break
    time.sleep(10)
print("run_id:", run_id, flush=True)
if not run_id:
    print("CLAIM-FAIL no run appeared")
    sys.exit(2)

deadline = time.time() + 900
status = conclusion = None
job_id = None
while time.time() < deadline:
    r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{run_id}")
    status, conclusion = r.get("status"), r.get("conclusion")
    print(f"poll {datetime.now(timezone.utc).strftime('%H:%M:%S')} status={status} conclusion={conclusion}", flush=True)
    if status == "completed":
        break
    time.sleep(90)  # 额度节流定案（作战室 04:16）：认领轮询 90 秒
if status != "completed":
    print("TIMEOUT run not completed in 900s")
    sys.exit(2)
print("run conclusion:", conclusion, flush=True)

jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{run_id}/jobs")
job_id = jobs["jobs"][0]["id"] if jobs.get("jobs") else None
print("job_id:", job_id, "job conclusion:", jobs["jobs"][0].get("conclusion") if jobs.get("jobs") else None, flush=True)

log = ""
for attempt in range(6):
    try:
        log = gh_api.get_job_logs(OWNER, REPO, job_id)
        break
    except Exception as e:
        print(f"log fetch attempt {attempt} failed: {type(e).__name__} {e}", flush=True)
        time.sleep(20)
out = os.path.join(HF, "studio-status-0412-claim.log")
open(out, "w").write(log or "")
print(f"log bytes: {len(log or '')} -> {out}", flush=True)
if MARK in (log or ""):
    print("MARKER CLAIMED", flush=True)
    print("STUDIO-END present:", "STUDIO-END" in log, flush=True)
    sys.exit(0 if "STUDIO-END" in log else 4)
print("MARKER NOT FOUND in log")
sys.exit(2)
