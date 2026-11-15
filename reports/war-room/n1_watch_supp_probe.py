#!/usr/bin/env python3
"""Supplementary read-only probe for n1 watch: correct log path + residual procs + eval.jsonl."""
import sys, time, base64, secrets, datetime
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import dispatch, req, get_job_logs

OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"
INSTANCE = "i-uf62k0o6zbe6j7zxnzg5"
MARKER = "N1WATCH-" + (sys.argv[1] if len(sys.argv) > 1 else secrets.token_hex(4).upper())

SCRIPT = r"""echo MARKER_PLACEHOLDER; date '+L20 %F %T %Z'
echo '== EVAL lines =='; grep -E 'EVAL' /root/carbon/data/rl3/snipe_n1.log | tail -12
echo '== eval.jsonl =='; cat /root/carbon/data/rl3/snipe_n1/eval.jsonl
echo '== traceback count =='; grep -c Traceback /root/carbon/data/rl3/snipe_n1.log || echo 0
echo '== traceback tail lines =='; grep -n Traceback /root/carbon/data/rl3/snipe_n1.log | tail -3
echo '== log tail =='; tail -5 /root/carbon/data/rl3/snipe_n1.log; stat -c 'log mtime %y size %s' /root/carbon/data/rl3/snipe_n1.log
echo '== residual procs =='; ps -eo pid,etime,args | grep -E 'ssh|scp|sftp|git' | grep -v grep
echo MARKER_PLACEHOLDER-END
""".replace("MARKER_PLACEHOLDER", MARKER)

b64 = base64.b64encode(SCRIPT.encode()).decode()
t0 = time.time()
print("MARKER", MARKER, flush=True)
r = dispatch(OWNER, REPO, WF, ref="main",
              inputs={"instance_id": INSTANCE, "region": "cn-shanghai", "command_b64": b64})
print("DISPATCH", r, flush=True)

run_id = None
for _ in range(30):
    runs = req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=10")
    for run in runs.get("workflow_runs", []):
        ct = datetime.datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")).timestamp()
        if ct >= t0 - 120:
            run_id = run["id"]
            print("RUN", run_id, run["status"], run["created_at"], flush=True)
            break
    if run_id:
        break
    time.sleep(5)
if not run_id:
    print("NO-RUN-FOUND"); sys.exit(2)

deadline = time.time() + 720
status = concl = None
while time.time() < deadline:
    run = req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{run_id}")
    status, concl = run["status"], run.get("conclusion")
    print(f"POLL {status} {concl} +{int(time.time()-t0)}s", flush=True)
    if status == "completed":
        break
    time.sleep(20)
print("FINAL", status, concl, flush=True)

jobs = req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{run_id}/jobs?per_page=5")
job = jobs["jobs"][0]
print("JOB", job["id"], job["name"], job.get("conclusion"), flush=True)
logs = get_job_logs(OWNER, REPO, job["id"])
print("===== LOGS BEGIN =====")
print(logs)
print("===== LOGS END =====")
