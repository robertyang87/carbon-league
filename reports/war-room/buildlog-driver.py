import gh_api, base64, time, os
MARK = "BUILOG16A"
os.chdir(os.path.expanduser("~/workspace"))
base = "goals/carbon-league-large-file-leaderboard-challenge/hidden_files/"
script = open(base + "buildlog.sh").read()
t0 = time.time()
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": "i-uf62k0o6zbe6j7zxnzg5", "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched at", time.strftime("%H:%M:%S"), flush=True)
deadline = time.time() + 300
seen_runs = set()
result = None
while time.time() < deadline:
    time.sleep(20)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=5")
    except Exception as e:
        print("list error:", e, flush=True); continue
    for run in r.get("workflow_runs", [])[:5]:
        rid = run["id"]
        if rid in seen_runs or run["status"] != "completed":
            continue
        seen_runs.add(rid)
        try:
            jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
        except Exception as e:
            print("jobs error:", e, flush=True); continue
        for j in jobs.get("jobs", []):
            try:
                log = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
            except Exception as e:
                print("log error:", e, flush=True); continue
            if isinstance(log, str) and MARK in log:
                print("CLAIMED run", rid, "job", j["id"], "status", run["status"], "conclusion", run.get("conclusion"), flush=True)
                i0 = log.find(MARK)
                out = base + "buildlog-full.log"
                open(out, "w").write(log)
                print("FULL LOG SAVED", out, "chars", len(log), flush=True)
                print(log[max(0, i0-200):i0+8000], flush=True)
                result = log
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time()-t0), "claimed" if result else "NOT-CLAIMED", flush=True)
