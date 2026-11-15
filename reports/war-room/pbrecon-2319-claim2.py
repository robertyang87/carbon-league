import gh_api, time
MARK = "PBRECON2319C"
LO, HI = "2026-11-12T15:17:00Z", "2026-11-12T15:22:30Z"
r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=100")
runs = [x for x in r.get("workflow_runs", []) if LO <= x["created_at"] <= HI]
runs.sort(key=lambda x: x["created_at"])
print("candidates:", [(x["id"], x["created_at"], x["status"]) for x in runs], flush=True)
found = None
for run in runs:
    rid = run["id"]
    if run["status"] != "completed":
        print("run", rid, "not completed, skip", flush=True); continue
    try:
        jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
    except Exception as e:
        print("run", rid, "jobs error:", e, flush=True); continue
    for j in jobs.get("jobs", []):
        log = None
        for attempt in range(8):
            try:
                log = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
                break
            except Exception as e:
                print("run", rid, "log retry", attempt, type(e).__name__, flush=True); time.sleep(3)
        if isinstance(log, str):
            if MARK in log:
                i0 = log.find(MARK)
                seg = log[max(0, i0 - 100): i0 + 80000]
                open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/pbrecon-2319-recovered.txt", "w").write(seg)
                print("CLAIMED run", rid, "job", j["id"], "seg chars", len(seg), flush=True)
                found = rid
                break
            else:
                print("run", rid, "log ok, no marker (", len(log), "chars )", flush=True)
    if found:
        break
print("RESULT", "claimed " + str(found) if found else "STILL-NOT-CLAIMED", flush=True)
