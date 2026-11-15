import gh_api, time, sys
MARK = "PBRECON2319C"
SINCE = "2026-11-12T15:16:00Z"
r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=50")
runs = [x for x in r.get("workflow_runs", []) if x["created_at"] >= SINCE]
print("runs in window:", len(runs), flush=True)
found = None
for run in runs:
    rid, st, cc = run["id"], run["status"], run.get("conclusion")
    print("run", rid, run["created_at"], st, cc, flush=True)
    if st != "completed":
        continue
    try:
        jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
    except Exception as e:
        print("  jobs error:", e, flush=True); continue
    for j in jobs.get("jobs", []):
        log = None
        for attempt in range(3):
            try:
                log = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
                break
            except Exception as e:
                print("  log retry", attempt, "error:", e, flush=True); time.sleep(5)
        if isinstance(log, str) and MARK in log:
            print("CLAIMED run", rid, "job", j["id"], flush=True)
            i0 = log.find(MARK)
            seg = log[max(0, i0 - 100): i0 + 60000]
            open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/pbrecon-2319-recovered.txt", "w").write(seg)
            print(seg[:12000], flush=True)
            found = rid
            break
    if found:
        break
print("RESULT", "claimed " + str(found) if found else "STILL-NOT-CLAIMED", flush=True)
