"""H24 read-only probe driver 2026-11-12 23:15 run. Marker: H24PROBE2326.
Target: spot-28 (i-uf65egu9fey7xf7tu8ft), cn-shanghai. READ-ONLY.
Usage: PYTHONPATH=/home/hatch/workspace python3 h24_probe_20261112_2326-driver.py
"""
import sys, base64, time
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

MARK = "H24PROBE2326"
SCRIPT = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24_probe_20261112_2326.sh"
LOG = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h24_probe_20261112_2326.log"
INST = "i-uf65egu9fey7xf7tu8ft"  # spot-28

script = open(SCRIPT).read()
assert MARK in script, "marker not in script"
t0 = time.time()
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": INST, "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched", INST, "at", time.strftime("%H:%M:%S"), flush=True)

deadline = time.time() + 420
seen = set()
result = None
while time.time() < deadline:
    time.sleep(15)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
    except Exception as e:
        print("list error:", e, flush=True)
        continue
    for run in r.get("workflow_runs", [])[:30]:
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
                open(LOG, "w").write(log)
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
