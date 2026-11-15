import sys, base64, time, datetime
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

# usage: m9xj260wr1_driver.py <l20_script.sh> <MARKER> <out_log> [deadline_s]
l20_path, MARK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
deadline_s = int(sys.argv[4]) if len(sys.argv) > 4 else 600
l20 = open(l20_path).read()
assert MARK in l20, "marker must appear in script"

t0 = time.time()
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": "i-uf62k0o6zbe6j7zxnzg5", "region": "cn-shanghai",
            "command_b64": base64.b64encode(l20.encode()).decode()})
print("dispatched", MARK, "at", time.strftime("%H:%M:%S"), flush=True)

min_created = datetime.datetime.utcfromtimestamp(t0 - 60).strftime("%Y-%m-%dT%H:%M:%SZ")
deadline = time.time() + deadline_s
seen = set()
result = None
while time.time() < deadline:
    time.sleep(20)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
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
    lines = result.splitlines()
    idx = [i for i, ln in enumerate(lines) if MARK in ln]
    lo = max(0, (idx[0] - 2) if idx else 0)
    print("\n".join(lines[lo:]), flush=True)
