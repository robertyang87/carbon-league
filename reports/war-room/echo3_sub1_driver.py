import base64, datetime, sys, time
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
MARKER = "E3SUB1" + time.strftime("%H%M%S")
INSTANCE = "i-uf62k0o6zbe6j7zxnzg5"  # gpu-l20-5
DGXK = "-i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
STUK = "-i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112"

script = f"""set +e
echo {MARKER}-BEGIN
hostname; date
echo '=== L20-PACK ==='
find /root/carbon -maxdepth 3 -name 'echo3_int8.py' 2>/dev/null | while read f; do echo "F $f $(stat -c %s "$f") $(md5sum "$f" | cut -d' ' -f1)"; done
echo '=== L20-GITHUB ==='
timeout -k 5 20 ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new -T git@github.com </dev/null 2>&1 | head -2
command -v gh >/dev/null && gh auth status 2>&1 | head -4 || echo 'no gh cli'
ls /root/.ssh/ 2>/dev/null
find /root/carbon -maxdepth 3 -name '.git' 2>/dev/null | head -4 | while read g; do d=$(dirname "$g"); echo "REPO $d"; git -C "$d" remote -v 2>/dev/null | head -2; done
echo '=== DGX ==='
timeout -k 30 240 ssh {DGXK} arm@10.211.43.155 'md5sum /home/arm/junyang/s26stage/echo3_int8.py; stat -c %s /home/arm/junyang/s26stage/echo3_int8.py; cd /home/arm/junyang/carbon-repo && git remote -v | head -2; timeout 40 git push --dry-run origin HEAD:refs/heads/echo3-dryrun 2>&1 | head -3; timeout -k 5 20 ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new -T git@github.com </dev/null 2>&1 | head -2; command -v gh >/dev/null && gh auth status 2>&1 | head -4 || echo no-gh-dgx; ls ~/.ssh/' </dev/null 2>&1 | head -30
echo "DGX-LEG-RC=$?"
echo '=== STUDIO ==='
timeout -k 30 180 ssh {STUK} arm@10.211.43.68 'ls -l /tmp/echo3_int8.py 2>/dev/null; md5 -q /tmp/echo3_int8.py 2>/dev/null; timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new -T git@github.com </dev/null 2>&1 | head -2; command -v gh >/dev/null && gh auth status 2>&1 | head -4 || echo no-gh-studio; ls ~/junyang 2>/dev/null' </dev/null 2>&1 | head -20
echo "STUDIO-LEG-RC=$?"
echo {MARKER}-END
"""
b64 = base64.b64encode(script.encode()).decode()
t0 = time.time()
t0iso = datetime.datetime.utcfromtimestamp(t0).strftime("%Y-%m-%dT%H:%M:%SZ")
print("dispatching marker", MARKER, flush=True)
r = gh_api.dispatch(OWNER, REPO, "cloud-assistant.yml", ref="main",
                    inputs={"instance_id": INSTANCE, "region": "cn-shanghai", "command_b64": b64})
print("dispatch resp:", r, flush=True)

deadline = t0 + 13 * 60
seen = set()
while time.time() < deadline:
    try:
        runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/cloud-assistant.yml/runs?per_page=30")
    except Exception as e:
        print("runs list error:", e, flush=True)
        time.sleep(45)
        continue
    for run in runs.get("workflow_runs", []):
        rid = run["id"]
        if rid in seen:
            continue
        if run.get("created_at", "") < t0iso:
            continue
        seen.add(rid)
        print("candidate run", rid, run.get("status"), run.get("conclusion"), flush=True)
        if run.get("status") != "completed":
            continue
        try:
            jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs")
        except Exception as e:
            print("jobs error:", e, flush=True)
            continue
        for job in jobs.get("jobs", []):
            try:
                log = gh_api.get_job_logs(OWNER, REPO, job["id"])
            except Exception as e:
                print("log error:", e, flush=True)
                continue
            if isinstance(log, tuple):
                log = log[0]
            if isinstance(log, bytes):
                log = log.decode("utf-8", "replace")
            if MARKER in log:
                print(f"=== CLAIMED run {rid} job {job['id']} ===", flush=True)
                i = log.find(MARKER + "-BEGIN")
                j = log.find(MARKER + "-END")
                seg = log[max(0, i - 200):(j + len(MARKER) + 4 if j >= 0 else i + 8000)]
                print(seg, flush=True)
                with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/echo3-sub1-claim.log", "w") as f:
                    f.write(log)
                print("=== PROBE DONE ===", flush=True)
                sys.exit(0)
    time.sleep(25)
print("PROBE TIMEOUT - no claim", flush=True)
sys.exit(1)
