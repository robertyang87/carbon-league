"""One-shot READ-ONLY diagnostic probe for the M7Bs v2 fix stage.

Situation: probe 003 (17:46:56 CST) showed the stage log stuck at the end of
the BUILD-DS output (games list, "(need 142)"), NO cnet2_train procs, GPU 0%,
and no LAUNCH / dataset-games / DONE / FAIL lines at all. The stage script
should have launched dual-arm training ~17:37-17:38. This probe only READS
state — it launches nothing, kills nothing, edits nothing.
"""
import gh_api, base64, time, os, sys

BASE = os.path.expanduser("~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/")
PMARK = "M7V2FXDIAG01"
t0 = time.time()
since_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 - 60))
os.chdir(os.path.expanduser("~/workspace"))

diag = """#!/bin/bash
MARK=__DIAG_MARK__
D=/home/arm/junyang/carbon/data
echo "$MARK DIAG-BEGIN $(date '+%F %T %Z')"
echo "--- stage script alive? ---"
pgrep -af 'm7v2fix' || echo NO-STAGE-PROC
echo "--- all python procs ---"
pgrep -af 'python' | head -20 || echo NO-PYTHON-PROC
echo "--- stage log: lines + last 6 ---"
wc -l $D/dq_logs/m7v2fix.log
tail -n 6 $D/dq_logs/m7v2fix.log | cut -c1-200
echo "--- stage log mtime ---"
stat -c '%y %n' $D/dq_logs/m7v2fix.log
echo "--- build log tail ---"
tail -n 3 $D/dq_logs/build_m7bsv2.log | cut -c1-300
echo "--- meta.json ---"
cat $D/ds_m7bs_v2/meta.json 2>&1 | cut -c1-300
echo "--- trainer logs exist? ---"
ls -la $D/dq_logs/m7bsv2r1.log $D/dq_logs/m7bsv2g1.log 2>&1
ls -la $D/dq_logs/m7bsv2r1.done $D/dq_logs/m7bsv2g1.done 2>&1
echo "--- trainer log tails (if any) ---"
tail -n 3 $D/dq_logs/m7bsv2r1.log 2>&1 | cut -c1-200
tail -n 3 $D/dq_logs/m7bsv2g1.log 2>&1 | cut -c1-200
echo "--- weights ---"
ls -la $D/weights/cnet2_m7bsv2* 2>&1
echo "--- gpu ---"
timeout 30 nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used --format=csv,noheader 2>&1
echo "$MARK DIAG-END"
"""

diag = diag.replace("__DIAG_MARK__", PMARK)
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": "i-uf62k0o6zbe6j7zxnzg5", "region": "cn-shanghai",
            "command_b64": base64.b64encode(diag.encode()).decode()})
print("dispatched diag", PMARK, flush=True)

seen = set()
deadline = time.time() + 300
while time.time() < deadline:
    time.sleep(20)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=15")
    except Exception as e:
        print("list error:", e, flush=True); continue
    for run in r.get("workflow_runs", [])[:15]:
        rid = run["id"]
        if rid in seen or run["status"] != "completed":
            continue
        if run.get("created_at", "") < since_iso:
            continue
        seen.add(rid)
        try:
            jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
        except Exception as e:
            print("jobs error:", e, flush=True); continue
        for j in jobs.get("jobs", []):
            try:
                lg = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
            except Exception as e:
                print("log error:", e, flush=True); continue
            if isinstance(lg, str) and PMARK in lg:
                print("CLAIMED run", rid, "job", j["id"], "conclusion", run.get("conclusion"), flush=True)
                open(BASE + "m7v2fix-diag-001.log", "w").write(lg)
                j0 = lg.find(PMARK)
                print(lg[max(0, j0 - 100):j0 + 8000], flush=True)
                sys.exit(0)
print("DIAG NOT-CLAIMED within 300s", flush=True)
sys.exit(1)
