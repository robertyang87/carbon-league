"""H29 weight transfer driver: dispatch transfer script to L20 via
cloud-assistant, claim run by unique marker only.
Usage: PYTHONPATH=/home/hatch/workspace python3 s29fxfer-2256-driver.py
"""
import sys, base64, time
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

MARK = "S29FXFER2256"
BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
SCRIPT = BASE + "/s29fxfer-2256.sh"
LOG = BASE + "/s29fxfer-2256.log"
INST = "i-uf62k0o6zbe6j7zxnzg5"  # L20
OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"

script = open(SCRIPT).read()
assert MARK in script
t0 = time.time()
gh_api.dispatch(OWNER, REPO, WF, ref="main",
    inputs={"instance_id": INST, "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched", INST, "at", time.strftime("%H:%M:%S"), flush=True)

deadline = time.time() + 900
seen = set()
result = None
while time.time() < deadline:
    time.sleep(15)
    try:
        r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30")
    except Exception as e:
        print("list error:", e, flush=True)
        continue
    for run in r.get("workflow_runs", [])[:30]:
        rid = run["id"]
        if rid in seen or run["status"] != "completed":
            continue
        seen.add(rid)
        try:
            jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs")
        except Exception:
            continue
        for j in jobs.get("jobs", []):
            try:
                log = gh_api.get_job_logs(OWNER, REPO, j["id"])
            except Exception:
                continue
            if isinstance(log, str) and MARK in log:
                print("CLAIMED run", rid, "conclusion", run.get("conclusion"), flush=True)
                open(LOG, "w").write(log)
                i0 = log.find(MARK + " L20-BEGIN")
                i1 = log.find(MARK + " L20-END")
                print(log[i0:i1 + len(MARK) + 10] if i0 >= 0 else log[:14000], flush=True)
                result = log
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time() - t0), "CLAIMED" if result else "NOT-CLAIMED", flush=True)
