#!/usr/bin/env python3
"""n1 watch probe driver: dispatch cloud-assistant.yml to L20, claim by run id, print logs.
Read-only on L20. Usage: python3 n1_watch_probe.py [marker_suffix]"""
import sys, json, time, base64, secrets
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import dispatch, req, get_job_logs

OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"
INSTANCE = "i-uf62k0o6zbe6j7zxnzg5"
MARKER = "N1WATCH-" + (sys.argv[1] if len(sys.argv) > 1 else secrets.token_hex(4).upper())

SCRIPT = r"""echo MARKER_PLACEHOLDER; date '+L20 %F %T %Z'
echo '== weights =='; ls -lt /root/carbon/data/weights/ | head -20
echo '== md5 known pool weights =='; md5sum /root/carbon/data/weights/cnet2_m3s2r1L.npz /root/carbon/data/weights/cnet2_m7bsr1L.npz /root/carbon/data/weights/cnet2_m7bsg1L.npz 2>&1
echo '== proc =='; pgrep -af "rl3/snipe_n1" || echo NO_PROC
echo '== gpu =='; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo '== snipe dir =='; ls -lt /root/carbon/data/rl3/snipe_n1/ | head -25
echo '== last4 ckpt =='; ls -td /root/carbon/data/rl3/snipe_n1/ckpt_* 2>/dev/null | head -4 | while read d; do stat -c '%y %n' "$d"; done
echo '== train.jsonl all rows =='; python3 -c "
import json
for line in open('/root/carbon/data/rl3/snipe_n1/train.jsonl'):
    line=line.strip()
    if not line: continue
    try: r=json.loads(line)
    except Exception: print('BADLINE', line[:120]); continue
    print({k: r.get(k) for k in ('it','pre_s','upd_s','winrate','kl','dlogp','kl_coef','amp')})
"
echo '== train.jsonl tail =='; tail -3 /root/carbon/data/rl3/snipe_n1/train.jsonl
echo '== EVAL lines =='; grep -E 'EVAL' /root/carbon/data/rl3/snipe_n1.log | tail -12
echo '== traceback count =='; grep -c Traceback /root/carbon/data/rl3/snipe_n1.log || echo 0
echo '== log tail =='; tail -5 /root/carbon/data/rl3/snipe_n1.log; stat -c 'log mtime %y size %s' /root/carbon/data/rl3/snipe_n1.log
echo '== pool go flag =='; ls -l /tmp/n1_pool_go.txt 2>&1
echo '== league/live =='; stat -c '%y %n' /root/carbon/data/rl3/snipe_n1/league.json /root/carbon/data/rl3/snipe_n1/live.json 2>&1
echo MARKER_PLACEHOLDER-END
""".replace("MARKER_PLACEHOLDER", MARKER)

b64 = base64.b64encode(SCRIPT.encode()).decode()
t0 = time.time()
print("MARKER", MARKER, flush=True)
r = dispatch(OWNER, REPO, WF, ref="main",
              inputs={"instance_id": INSTANCE, "region": "cn-shanghai", "command_b64": b64})
print("DISPATCH", r, flush=True)

# find the new run: list workflow runs, pick newest created after t0
run_id = None
for _ in range(30):
    runs = req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=10")
    for run in runs.get("workflow_runs", []):
        created = run["created_at"]
        # created_at ISO UTC; dispatch happened just now
        import datetime
        ct = datetime.datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
        if ct >= t0 - 120:
            run_id = run["id"]
            print("RUN", run_id, run["status"], run["created_at"], flush=True)
            break
    if run_id:
        break
    time.sleep(5)
if not run_id:
    print("NO-RUN-FOUND"); sys.exit(2)

# poll to completion (max ~12 min)
deadline = time.time() + 720
status = None
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
