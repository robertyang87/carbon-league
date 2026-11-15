"""C1 diagnostic pull driver: dispatch pull script to L20 via
cloud-assistant, claim run by unique marker only, dump raw log sections.
Usage: PYTHONPATH=/home/hatch/workspace python3 s29fg1diag-2346-driver.py
"""
import sys, base64, time
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

MARK = "S29FG1DIAG2346"
BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
SCRIPT = BASE + "/s29fg1diag-2346.sh"
LOG = BASE + "/s29fg1diag-2346.log"
INST = "i-uf62k0o6zbe6j7zxnzg5"  # L20
OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"

script = open(SCRIPT).read()
assert MARK in script
t0 = time.time()
gh_api.dispatch(OWNER, REPO, WF, ref="main",
    inputs={"instance_id": INST, "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched", INST, "at", time.strftime("%H:%M:%S"), flush=True)

deadline = time.time() + 600
seen = set()
result = None
while time.time() < deadline:
    time.sleep(15)
    try:
        r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30")
    except Exception as e:
        print("list error:", e, flush=True)
        continue
    for run in r.get("workflow_runs", [])[:30]:
        rid = run["id"]
        if rid in seen or run["status"] != "completed":
            continue
        seen.add(rid)
        try:
            jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs")
        except Exception:
            continue
        for j in jobs.get("jobs", []):
            try:
                log = gh_api.get_job_logs(OWNER, REPO, j["id"])
            except Exception:
                continue
            if isinstance(log, str) and MARK in log:
                print("CLAIMED run", rid, "conclusion", run.get("conclusion"), flush=True)
                open(LOG, "w").write(log)
                i0 = log.find("--- SUMMARIES RAW ---")
                i1 = log.find("--- B64 fabler1_echo2 ---")
                seg = log[i0:i1] if i0 >= 0 else log[:8000]
                # strip Actions timestamps for readability
                out = []
                for ln in seg.splitlines():
                    parts = ln.split(" ", 1)
                    out.append(parts[1] if len(parts) > 1 else ln)
                print("\n".join(out), flush=True)
                b = log.find("--- B64 fabler1_echo2 ---")
                print("B64 section present:", b >= 0, "| log chars:", len(log), flush=True)
                if b >= 0:
                    tail = log[b:b+400]
                    print("B64 head raw:", tail[:400], flush=True)
                result = log
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time() - t0), "CLAIMED" if result else "NOT-CLAIMED", flush=True)
