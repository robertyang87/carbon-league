"""DGX status probe driver 0428: dispatch cloud-assistant run, claim by marker.
Marker-only claim (never time-window), per 2026-11-12 cross-talk lesson.
90s polling per 2026-11-13 04:16 quota decision; 10 min cap.
Skips log fetch for jobs with zero steps (04:13 lesson: 0-step failures
have no log blob; fetching them burns quota)."""
import base64, time, os, sys

sys.path.insert(0, os.path.expanduser("~/workspace"))
import gh_api

MARK = "DGXSTATUS0513A"
HF = os.path.expanduser(
    "~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files")
LOG = os.path.join(HF, "dgx-status-0513.log")
SCRIPT = open(os.path.join(HF, "dgx-status-0513.sh")).read()

t0 = time.time()
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": "i-uf62k0o6zbe6j7zxnzg5", "region": "cn-shanghai",
            "command_b64": base64.b64encode(SCRIPT.encode()).decode()})
print("dispatched at", time.strftime("%H:%M:%S"), flush=True)
deadline = time.time() + 600
tried = {}
result = None
seen_failed = []
while time.time() < deadline:
    time.sleep(90)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
    except Exception as e:
        print("list error:", e, flush=True)
        continue
    for run in r.get("workflow_runs", []):
        rid = run["id"]
        if run["status"] != "completed":
            continue
        if run["created_at"] < time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 - 120)):
            continue
        if tried.get(rid, 0) >= 2:
            continue
        tried[rid] = tried.get(rid, 0) + 1
        print("trying run", rid, "round", tried[rid], "conclusion", run.get("conclusion"), flush=True)
        try:
            jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
        except Exception as e:
            print("jobs error:", e, flush=True)
            continue
        for j in jobs.get("jobs", []):
            steps = j.get("steps") or []
            if not steps:
                print(f"  job {j['id']} has 0 steps, skipping log fetch", flush=True)
                seen_failed.append((rid, j["id"]))
                continue
            try:
                log = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
            except Exception as e:
                print("log error:", e, flush=True)
                continue
            log = log if isinstance(log, str) else ""
            print(f"  job {j['id']} log bytes={len(log)}", flush=True)
            if MARK in log:
                print("CLAIMED run", rid, "job", j["id"], "conclusion", run.get("conclusion"), flush=True)
                i0 = log.find(MARK)
                print(log[max(0, i0 - 200):i0 + 12000], flush=True)
                with open(LOG, "w") as f:
                    f.write(log)
                result = log
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time() - t0), "claimed" if result else "NOT-CLAIMED", flush=True)
print("zero-step jobs seen:", seen_failed, flush=True)
