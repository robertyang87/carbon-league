"""H28 K4 probe launch driver: dispatch launch script to spot-28 via
cloud-assistant (GitHub Actions), claim the run by unique marker only.
Usage: PYTHONPATH=/home/hatch/workspace python3 s28k4launch-2252-driver.py
"""
import sys, base64, time
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

MARK = "S28K4LAUNCH2252"
BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
SCRIPT = BASE + "/s28k4launch-2252.sh"
LOG = BASE + "/s28k4launch-2252.log"
INST = "i-uf65egu9fey7xf7tu8ft"  # spot-28
OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"

script = open(SCRIPT).read()
assert MARK in script, "marker not in script"
t0 = time.time()
gh_api.dispatch(OWNER, REPO, WF, ref="main",
    inputs={"instance_id": INST, "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched", INST, "at", time.strftime("%H:%M:%S"), flush=True)

deadline = time.time() + 600
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
                i0 = log.find(MARK + "_BEGIN")
                i1 = log.find(MARK + "_END")
                print(log[i0:i1 + len(MARK) + 4] if i0 >= 0 else log[:12000], flush=True)
                result = log
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time() - t0), "CLAIMED" if result else "NOT-CLAIMED", flush=True)
