import datetime, sys, time
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
MARKER = "ECHO3SUBMIT-RUN1"
t0 = time.time()
t0iso = datetime.datetime.utcfromtimestamp(t0).strftime("%Y-%m-%dT%H:%M:%SZ")
print("dispatching ladder-submit at", t0iso, flush=True)
r = gh_api.dispatch(OWNER, REPO, "ladder-submit.yml", ref="main", inputs={})
print("dispatch resp:", r, flush=True)

deadline = t0 + 13 * 60
seen = set()
target = None
while time.time() < deadline:
    try:
        runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/ladder-submit.yml/runs?per_page=10")
    except Exception as e:
        print("runs list error:", e, flush=True)
        time.sleep(40)
        continue
    for run in runs.get("workflow_runs", []):
        rid = run["id"]
        if run.get("created_at", "") < t0iso:
            continue
        if target is None:
            target = rid
            print("target run:", rid, run.get("status"), run.get("conclusion"), flush=True)
        if rid != target:
            continue
        print("run", rid, run.get("status"), run.get("conclusion"), flush=True)
        if run.get("status") != "completed":
            continue
        try:
            jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs")
        except Exception as e:
            print("jobs error:", e, flush=True)
            continue
        for job in jobs.get("jobs", []):
            try:
                log = gh_api.get_job_logs(OWNER, REPO, job["id"])
            except Exception as e:
                print("log error:", e, flush=True)
                continue
            if isinstance(log, tuple):
                log = log[0]
            if isinstance(log, bytes):
                log = log.decode("utf-8", "replace")
            with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/echo3-submit-claim.log", "w") as f:
                f.write(log)
            i = log.find("Verify pack bytes")
            seg = log[i - 200:] if i >= 0 else log[-9000:]
            print(f"=== SUBMIT RUN {rid} conclusion={run.get('conclusion')} job={job['id']} ===", flush=True)
            print(seg[-9000:], flush=True)
            print("=== SUBMIT TRIP DONE ===", flush=True)
            sys.exit(0)
        sys.exit(0)
    time.sleep(25)
print("SUBMIT CLAIM TIMEOUT target=", target, flush=True)
sys.exit(1)
