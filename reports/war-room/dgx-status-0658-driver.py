"""DGX status probe trip 06:58. Dispatch cloud-assistant.yml to L20, then claim
the run by the unique marker DGXSTATUS0658A (per_page=30, created_at cutoff).
Logs everything to dgx-status-0658-driver.log. Read-only probe."""
import sys, os, time, base64
sys.path.insert(0, os.path.expanduser("~/workspace"))
import gh_api

MARK = "DGXSTATUS0658A"
HF = os.path.expanduser("~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files")
LOGF = os.path.join(HF, "dgx-status-0658-driver.log")

def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOGF, "a") as f:
        f.write(line + "\n")

script = open(os.path.join(HF, "dgx-status-0658.sh")).read()
t0 = time.time()
log("dispatching at", time.strftime("%H:%M:%S"), "marker", MARK)
try:
    gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
        inputs={"instance_id": "i-uf62k0o6zbe6j7zxnzg5", "region": "cn-shanghai",
                "command_b64": base64.b64encode(script.encode()).decode()})
    log("dispatch OK")
except Exception as e:
    log("dispatch FAILED:", repr(e))
    sys.exit(2)

cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 - 120))
deadline = time.time() + 600
seen, tried = set(), {}
result = None
while time.time() < deadline:
    time.sleep(30)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
    except Exception as e:
        log("list error:", repr(e))
        if "403" in str(e):
            time.sleep(60)
        continue
    for run in r.get("workflow_runs", []):
        rid = run["id"]
        if rid in seen or run["status"] != "completed":
            continue
        if run["created_at"] < cutoff:
            continue
        if tried.get(rid, 0) >= 4:
            continue
        tried[rid] = tried.get(rid, 0) + 1
        log("trying run", rid, "round", tried[rid], "conclusion", run.get("conclusion"),
            "created_at", run.get("created_at"))
        try:
            jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
        except Exception as e:
            log("jobs error:", repr(e))
            continue
        for j in jobs.get("jobs", []):
            try:
                joblog = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
            except Exception as e:
                log("log error job", j["id"], repr(e))
                continue
            log("  job", j["id"], "log bytes=", len(joblog) if isinstance(joblog, str) else "?")
            if isinstance(joblog, str) and MARK in joblog:
                i0 = joblog.find(MARK)
                log("CLAIMED run", rid, "job", j["id"], "conclusion", run.get("conclusion"))
                log(joblog[max(0, i0 - 200):i0 + 12000])
                with open(os.path.join(HF, "dgx-status-0658.log"), "w") as f:
                    f.write(joblog)
                result = joblog
                seen.add(rid)
                break
        if result:
            break
    if result:
        break
log("RESULT:", "CLAIMED" if result else "NOT-CLAIMED (no run with marker in window)")
