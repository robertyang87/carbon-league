#!/usr/bin/env python3
"""Studio status probe 0357 driver: dispatch cloud-assistant.yml probe to L20,
poll workflow runs, claim the run whose job log contains our unique marker,
save the log. Read-only probe; no Studio-side changes."""
import sys, base64, time, datetime, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

MARK = "STUDIOSTAT0357HAKIMI"
HF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
script = open(f"{HF}/studio-status-0357.sh").read()
t0 = time.time()
dispatch_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": "i-uf62k0o6zbe6j7zxnzg5", "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched at", time.strftime("%H:%M:%S"), "dispatch_ts", dispatch_ts, flush=True)

deadline = time.time() + 480
seen = set()
while time.time() < deadline:
    time.sleep(45)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
    except Exception as e:
        print("list error:", e, flush=True); continue
    for run in r.get("workflow_runs", []):
        rid = run["id"]
        if rid in seen:
            continue
        if run.get("created_at", "") < dispatch_ts:
            continue
        if run["status"] != "completed":
            continue
        seen.add(rid)
        # fetch jobs, find the job whose log carries our marker
        try:
            jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs?per_page=10")
        except Exception as e:
            print("jobs error:", rid, e, flush=True); continue
        for job in jobs.get("jobs", []):
            try:
                log = gh_api.get_job_logs("robertyang87", "carbon-league", job["id"])
            except Exception as e:
                print("log error:", job["id"], e, flush=True); continue
            if MARK in log:
                out = f"{HF}/studio-status-0357.log"
                open(out, "w").write(log)
                print("CLAIMED run", rid, "job", job["id"], "conclusion", run.get("conclusion"),
                      "elapsed", round(time.time() - t0), "s ->", out, flush=True)
                sys.exit(0)
        print("completed run", rid, "not ours (no marker)", flush=True)
print("TIMEOUT no claim after", round(time.time() - t0), "s; seen:", sorted(seen), flush=True)
sys.exit(1)
