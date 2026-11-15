import gh_api, base64, time, os, sys
MARK = "SPOT25E5K7"
INSTANCE = "i-uf62k0o6zbe6j7zxnzg5"
os.chdir(os.path.expanduser("~/workspace"))
base = "goals/carbon-league-large-file-leaderboard-challenge/hidden_files/"
script = f"""#!/bin/bash
MARK="{MARK}"
echo "MARKER $MARK START"
date '+%F %T %Z'
for i in 1 2 3 4 5 6 7; do
  OUT=$(timeout 90 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'tail -25 /home/arm/s25tmp/xfer.log' 2>&1)
  echo "--- poll $i ---"
  echo "$OUT"
  if echo "$OUT" | grep -qE 'XFER-DONE|XFER-FAIL'; then echo "POLL-TERMINAL"; break; fi
  sleep 55
done
echo "MARKER $MARK END"
"""
t0 = time.time()
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": INSTANCE, "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched at", time.strftime("%H:%M:%S"), flush=True)
deadline = time.time() + 700
seen_runs = set()
result = None
while time.time() < deadline:
    time.sleep(20)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=8")
    except Exception as e:
        print("list error:", e, flush=True); continue
    for run in r.get("workflow_runs", [])[:8]:
        rid = run["id"]
        if rid in seen_runs or run["status"] != "completed":
            continue
        seen_runs.add(rid)
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
                out = base + "spot25-e-full.log"
                open(out, "w").write(log)
                print("FULL LOG SAVED", out, "chars", len(log), flush=True)
                i0 = log.find("MARKER " + MARK + " START")
                print(log[i0:i0+10000] if i0 >= 0 else log[:10000], flush=True)
                result = log
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time()-t0), "claimed" if result else "NOT-CLAIMED", flush=True)
