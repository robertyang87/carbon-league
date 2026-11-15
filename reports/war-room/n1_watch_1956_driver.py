import sys, time, base64, datetime
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
MARK = "N1WATCH-1956340F00B7"
script = open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/n1_watch_1956.sh").read()
t0 = time.time()
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": "i-uf62k0o6zbe6j7zxnzg5", "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched at", time.strftime("%H:%M:%S"), "MARK", MARK, flush=True)
cutoff = datetime.datetime.fromtimestamp(t0 - 60, tz=datetime.timezone.utc)
deadline = time.time() + 480
result = None
while time.time() < deadline:
    time.sleep(20)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
    except Exception as e:
        print("list error:", e, flush=True); continue
    for run in r.get("workflow_runs", []):
        ct = datetime.datetime.fromisoformat(run["created_at"].replace("Z","+00:00"))
        if ct < cutoff or run["status"] != "completed":
            continue
        try:
            jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{run['id']}/jobs")
        except Exception as e:
            print("jobs error:", e, flush=True); continue
        for j in jobs.get("jobs", []):
            try:
                log = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
            except Exception as e:
                print("log error:", e, flush=True); continue
            if isinstance(log, str) and MARK in log:
                print("CLAIMED run", run["id"], "job", j["id"], "conclusion", run.get("conclusion"), flush=True)
                i0 = log.find(MARK)
                print(log[max(0, i0-200):i0+16000], flush=True)
                with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/n1_watch_1956_run.log", "w") as f:
                    f.write(log)
                result = log
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time()-t0), "claimed" if result else "NOT-CLAIMED", flush=True)
