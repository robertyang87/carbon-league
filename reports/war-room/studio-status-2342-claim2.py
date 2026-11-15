import sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
MARK = "STUDIOSTAT2342IT320"
rid = 43251320363
jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
print("jobs:", [(j["id"], j.get("name"), j.get("status"), j.get("conclusion")) for j in jobs.get("jobs", [])])
for j in jobs.get("jobs", []):
    try:
        log = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
    except Exception as e:
        print("log error job", j["id"], ":", e)
        continue
    print("job", j["id"], "log len", len(log) if isinstance(log, str) else type(log))
    if isinstance(log, str) and MARK in log:
        i0 = log.find(MARK)
        print(log[max(0, i0-200):i0+16000])
        print("=== MARKER FOUND ===")
    else:
        print("marker NOT in this job log")
