"""Generic cloud-assistant driver: dispatch a script, claim by unique marker.
usage: k4v2_driver.py <script.sh> <MARKER> <out_log> [timeout_s] [tail_chars]
"""
import sys, os, base64, time, datetime
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

script_path, MARK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
TIMEOUT = int(sys.argv[4]) if len(sys.argv) > 4 else 540
TAIL = int(sys.argv[5]) if len(sys.argv) > 5 else 14000
script = open(script_path).read()
t0 = time.time()
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": "i-uf62k0o6zbe6j7zxnzg5", "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched", MARK, "at", time.strftime("%H:%M:%S"), flush=True)
min_created = datetime.datetime.utcfromtimestamp(t0 - 60).strftime("%Y-%m-%dT%H:%M:%SZ")
deadline = time.time() + TIMEOUT
seen, result = set(), None
while time.time() < deadline:
    time.sleep(15)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=10")
    except Exception as e:
        print("list error:", e, flush=True); continue
    for run in r.get("workflow_runs", []):
        rid = run["id"]
        if rid in seen or run["status"] != "completed":
            continue
        if run.get("created_at", "") < min_created:
            continue
        seen.add(rid)
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
                open(OUT, "w").write(log)
                result = log
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time() - t0), "CLAIMED" if result else "NOT-CLAIMED", flush=True)
if result:
    i0 = result.find(MARK)
    print(result[max(0, i0 - 200):i0 + TAIL], flush=True)
