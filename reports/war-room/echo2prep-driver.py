import gh_api, base64, time, os, sys

# usage: echo2prep-driver.py <script.sh> <instance_id> <MARK> <outlog> [deadline_s]
script_path, instance_id, MARK, outlog = sys.argv[1:5]
deadline_s = int(sys.argv[5]) if len(sys.argv) > 5 else 600
os.chdir(os.path.expanduser("~/workspace"))
script = open(script_path).read()
assert MARK in script, "marker missing from script"
t0 = time.time()
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": instance_id, "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched", script_path, "->", instance_id, "at", time.strftime("%H:%M:%S"), flush=True)
deadline = time.time() + deadline_s
seen_runs = set()
result = None
while time.time() < deadline:
    time.sleep(15)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=10")
    except Exception as e:
        print("list error:", e, flush=True); continue
    for run in r.get("workflow_runs", [])[:10]:
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
                print("CLAIMED run", rid, "job", j["id"], "conclusion", run.get("conclusion"), flush=True)
                open(outlog, "w").write(log)
                print("FULL LOG SAVED", outlog, "chars", len(log), flush=True)
                i0 = log.find(MARK)
                print(log[max(0, i0 - 100):i0 + 12000], flush=True)
                result = log
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time() - t0), "claimed" if result else "NOT-CLAIMED", flush=True)
