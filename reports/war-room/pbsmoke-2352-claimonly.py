import gh_api, time
MARK = "PBSMOKE2352B"
SINCE = "2026-11-12T15:44:30Z"
deadline = time.time() + 1500
fetched_ok = set()
result = None
while time.time() < deadline:
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
    except Exception as e:
        print("list error:", e, flush=True); time.sleep(25); continue
    for run in r.get("workflow_runs", []):
        rid = run["id"]
        if rid in fetched_ok or run["status"] != "completed" or run["created_at"] < SINCE:
            continue
        try:
            jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
        except Exception as e:
            print("jobs error:", e, flush=True); continue
        for j in jobs.get("jobs", []):
            try:
                log = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
            except Exception as e:
                print("log error (will retry):", type(e).__name__, flush=True); continue
            fetched_ok.add(rid)
            if isinstance(log, str) and MARK in log:
                print("CLAIMED run", rid, "job", j["id"], "conclusion", run.get("conclusion"), flush=True)
                i0 = log.find(MARK)
                seg = log[max(0, i0 - 200):i0 + 45000]
                open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/pbsmoke-2352-recovered.txt", "w").write(seg)
                print(seg, flush=True)
                result = rid
                break
        if result:
            break
    if result:
        break
    time.sleep(25)
print("RESULT", "claimed " + str(result) if result else "STILL-NOT-CLAIMED", flush=True)
