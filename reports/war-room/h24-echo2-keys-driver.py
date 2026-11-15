"""Fresh parametrized driver: dispatch a cloud-assistant run, claim by marker.
Usage: python3 h24-echo2-keys-driver.py <script.sh> <instance_id> <marker> <outlog> [timeout_s]
"""
import sys, base64, time
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/"
script_path, INST, MARK, outlog = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
tmo = int(sys.argv[5]) if len(sys.argv) > 5 else 420

script = open(script_path).read()
assert MARK in script, "marker not in script"
t0 = time.time()
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": INST, "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched", INST, MARK, "at", time.strftime("%H:%M:%S"), flush=True)

deadline = time.time() + tmo
seen = set()
result = None
while time.time() < deadline:
    time.sleep(15)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=10")
    except Exception as e:
        print("list error:", e, flush=True)
        continue
    for run in r.get("workflow_runs", [])[:10]:
        rid = run["id"]
        if rid in seen or run["status"] != "completed":
            continue
        seen.add(rid)
        try:
            jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
        except Exception:
            continue
        for j in jobs.get("jobs", []):
            try:
                log = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
            except Exception:
                continue
            if isinstance(log, str) and MARK in log:
                print("CLAIMED run", rid, "conclusion", run.get("conclusion"), flush=True)
                open(outlog, "w").write(log)
                i0 = log.find(MARK + "_BEGIN")
                i1 = log.find(MARK + "_END")
                print(log[i0:i1 + len(MARK) + 4] if i0 >= 0 else log[:8000], flush=True)
                result = log
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time() - t0), "CLAIMED" if result else "NOT-CLAIMED", flush=True)
