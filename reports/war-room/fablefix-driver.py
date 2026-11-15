import sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api, base64, time

script_path, MARK = sys.argv[1], sys.argv[2]
POLL_ONLY = len(sys.argv) > 3 and sys.argv[3] == "poll"
script = open(script_path).read()
t0 = time.time()
if POLL_ONLY:
    t0 = t0 - 240  # the dispatch already happened ~4 min ago; still exclude older runs
    print("poll-only for", MARK, flush=True)
else:
    gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
        inputs={"instance_id": "i-uf62k0o6zbe6j7zxnzg5", "region": "cn-shanghai",
                "command_b64": base64.b64encode(script.encode()).decode()})
    print("dispatched", MARK, "at", time.strftime("%H:%M:%S"), flush=True)
import datetime
min_created = datetime.datetime.utcfromtimestamp(t0 - 60).strftime("%Y-%m-%dT%H:%M:%SZ")
deadline = time.time() + 570
seen_runs = set()
result = None
while time.time() < deadline:
    time.sleep(20)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=10")
    except Exception as e:
        print("list error:", e, flush=True); continue
    for run in r.get("workflow_runs", []):
        rid = run["id"]
        if rid in seen_runs or run["status"] != "completed":
            continue
        if run.get("created_at", "") < min_created:
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
                print(log, flush=True)
                result = log
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time()-t0), "claimed" if result else "NOT-CLAIMED", flush=True)
