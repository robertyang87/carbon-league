import gh_api, base64, time, os, sys
MARK = "SPOT25F3K7"
INSTANCE = "i-uf62k0o6zbe6j7zxnzg5"
os.chdir(os.path.expanduser("~/workspace"))
base = "goals/carbon-league-large-file-leaderboard-challenge/hidden_files/"
repro_b64 = base64.b64encode(open(base + "spot25-repro.py", "rb").read()).decode()
dgx = f"""KEY=/home/arm/.ssh/id_ed25519_carbon_relay
H24=47.117.16.90
S25=47.116.17.244
SCP="scp -i $KEY -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
SSH="ssh -n -i $KEY -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
echo "--- PULL S64 ---"
$SCP root@$H24:/root/carbon/data/weights/cnet2_s64.npz /home/arm/s25tmp/w/cnet2_s64.npz </dev/null && md5sum /home/arm/s25tmp/w/cnet2_s64.npz || echo "FAIL-PULL"
echo "--- PUSH S64 ---"
$SCP /home/arm/s25tmp/w/cnet2_s64.npz root@$S25:/root/carbon/data/weights/cnet2_s64.npz </dev/null && echo PUSH-OK || echo "FAIL-PUSH"
echo "--- MD5 ON S25 ---"
$SSH root@$S25 'md5sum /root/carbon/data/weights/cnet2_s64.npz'
echo "--- REPRO ON S25 ---"
echo {repro_b64} | base64 -d > /tmp/s25repro.py
$SCP /tmp/s25repro.py root@$S25:/tmp/s25repro.py </dev/null
$SSH root@$S25 'cd /root/carbon/code && timeout 180 ../.venv/bin/python /tmp/s25repro.py 2>&1 | tail -12'
"""
script = f"""#!/bin/bash
MARK="{MARK}"
echo "MARKER $MARK START"
date '+%F %T %Z'
timeout 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
{dgx}
DGXEOF
echo "DGX-SSH-RC=$?"
echo "MARKER $MARK END"
"""
t0 = time.time()
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": INSTANCE, "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched at", time.strftime("%H:%M:%S"), flush=True)
deadline = time.time() + 500
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
                out = base + "spot25-f7-full.log"
                open(out, "w").write(log)
                print("FULL LOG SAVED", out, "chars", len(log), flush=True)
                i0 = log.find("MARKER " + MARK + " START")
                print(log[i0:i0+8000] if i0 >= 0 else log[:8000], flush=True)
                result = log
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time()-t0), "claimed" if result else "NOT-CLAIMED", flush=True)
