import gh_api, base64, time, os
MARK = "PBSMOKE2345A"
os.chdir(os.path.expanduser("~/workspace"))
script = open("goals/carbon-league-large-file-leaderboard-challenge/hidden_files/pbsmoke-2345.sh").read()
t0 = time.time()
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": "i-uf62k0o6zbe6j7zxnzg5", "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched at", time.strftime("%H:%M:%S"), flush=True)
deadline = time.time() + 1750
fetched_ok = set()
result = None
while time.time() < deadline:
    time.sleep(25)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
    except Exception as e:
        print("list error:", e, flush=True); continue
    for run in r.get("workflow_runs", []):
        rid = run["id"]
        if rid in fetched_ok or run["status"] != "completed":
            continue
        if run["created_at"] < time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 - 120)):
            continue
        try:
            jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
        except Exception as e:
            print("jobs error:", e, flush=True); continue
        for j in jobs.get("jobs", []):
            try:
                log = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
            except Exception as e:
                print("log error (will retry):", e, flush=True); continue
            fetched_ok.add(rid)
            if isinstance(log, str) and MARK in log:
                print("CLAIMED run", rid, "job", j["id"], "conclusion", run.get("conclusion"), flush=True)
                i0 = log.find(MARK)
                seg = log[max(0, i0 - 200):i0 + 45000]
                open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/pbsmoke-2345-recovered.txt", "w").write(seg)
                print(seg, flush=True)
                result = log
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time() - t0), "claimed" if result else "NOT-CLAIMED", flush=True)
