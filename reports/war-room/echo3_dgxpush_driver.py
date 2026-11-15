import base64, datetime, sys, time
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
MARKER = "E3DGX1" + time.strftime("%H%M%S")
INSTANCE = "i-uf62k0o6zbe6j7zxnzg5"  # gpu-l20-5
DGXK = "-i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
EXP_MD5 = "7b7620499411a088cf4dc17c0e929bd7"
EXP_SIZE = "4576790"

script = f"""set +e
echo {MARKER}-BEGIN
hostname; date
timeout -k 30 700 ssh {DGXK} arm@10.211.43.155 'bash -s' <<'DGXEOF'
set +e
MARK=__MARKER__
EXP_MD5=__EXP_MD5__; EXP_SIZE=__EXP_SIZE__
SRC=/home/arm/junyang/s26stage/echo3_int8.py
echo "DGX-BEGIN $(hostname) $(date '+%F %T %Z')"
ls -l "$SRC" || {{ echo "RESULT=SRC-MISSING"; echo "DGX-END"; exit 0; }}
M=$(md5sum "$SRC" | cut -d' ' -f1); S=$(stat -c %s "$SRC")
echo "SRC md5=$M size=$S"
if [ "$M" != "$EXP_MD5" ] || [ "$S" != "$EXP_SIZE" ]; then echo "RESULT=SRC-MISMATCH"; echo "DGX-END"; exit 0; fi
cd /home/arm/junyang/carbon-repo || {{ echo "RESULT=NO-CLONE"; echo "DGX-END"; exit 0; }}
timeout 120 git fetch origin main 2>&1 | tail -2
BASE=$(git rev-parse origin/main); echo "BASE=$BASE"
WT=/tmp/e3wt-$$
git worktree remove --force "$WT" 2>/dev/null; rm -rf "$WT"
git worktree add --detach "$WT" origin/main 2>&1 | tail -1
cd "$WT" || {{ echo "RESULT=WT-FAILED"; echo "DGX-END"; exit 0; }}
mkdir -p code/packs
cp "$SRC" code/packs/echo3_int8.py
M2=$(md5sum code/packs/echo3_int8.py | cut -d' ' -f1); echo "COPY-MD5=$M2"
if [ "$M2" != "$EXP_MD5" ]; then echo "RESULT=COPY-MISMATCH"; cd /home/arm/junyang/carbon-repo; git worktree remove --force "$WT"; echo "DGX-END"; exit 0; fi
git add code/packs/echo3_int8.py
git -c user.name="carbon-ci" -c user.email="ci@users.noreply.github.com" commit -m "data: add Echo3 int8 pack (m3s2r1, md5 7b762049) for ladder submission" 2>&1 | tail -1
BLOB=$(git rev-parse HEAD:code/packs/echo3_int8.py); BSZ=$(git cat-file -s "$BLOB")
echo "BLOB=$BLOB SIZE=$BSZ HEAD=$(git rev-parse HEAD)"
timeout 300 git push origin HEAD:main 2>&1 | tail -3
TIP=$(git ls-remote origin refs/heads/main | cut -f1)
echo "REMOTE-MAIN-TIP=$TIP"
if [ "$TIP" = "$(git rev-parse HEAD)" ]; then echo "RESULT=PUSHED-MAIN head=$TIP blob=$BLOB size=$BSZ"; else echo "RESULT=PUSH-UNCONFIRMED tip=$TIP head=$(git rev-parse HEAD)"; fi
cd /home/arm/junyang/carbon-repo && git worktree remove --force "$WT"
echo "DGX-END"
DGXEOF
echo "DGX-LEG-RC=$?"
echo {MARKER}-END
"""
script = script.replace("__MARKER__", MARKER).replace("__EXP_MD5__", EXP_MD5).replace("__EXP_SIZE__", EXP_SIZE)
b64 = base64.b64encode(script.encode()).decode()
t0 = time.time()
t0iso = datetime.datetime.utcfromtimestamp(t0).strftime("%Y-%m-%dT%H:%M:%SZ")
print("dispatching marker", MARKER, flush=True)
r = gh_api.dispatch(OWNER, REPO, "cloud-assistant.yml", ref="main",
                    inputs={"instance_id": INSTANCE, "region": "cn-shanghai", "command_b64": b64})
print("dispatch resp:", r, flush=True)

deadline = t0 + 14 * 60
seen = set()
claimed = False
while time.time() < deadline:
    # Independent effect check: pack on main with exact size = success regardless of log claim.
    try:
        f = gh_api.get_file(OWNER, REPO, "code/packs/echo3_int8.py", ref="main")
        if isinstance(f, dict) and f.get("size") == 4576790:
            print(f"=== EFFECT CONFIRMED VIA API: size={f.get('size')} sha={f.get('sha')} ===", flush=True)
            sys.exit(0)
    except Exception:
        pass
    try:
        runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/cloud-assistant.yml/runs?per_page=30")
    except Exception as e:
        print("runs list error:", e, flush=True)
        time.sleep(40)
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
                claimed = True
                print(f"=== CLAIMED run {rid} job {job['id']} ===", flush=True)
                i = log.find(MARKER + "-BEGIN")
                j = log.find(MARKER + "-END")
                seg = log[max(0, i - 200):(j + len(MARKER) + 4 if j >= 0 else i + 9000)]
                print(seg, flush=True)
                with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/echo3-dgxpush-claim.log", "w") as f:
                    f.write(log)
                print("=== DGX PUSH TRIP CLAIMED ===", flush=True)
                sys.exit(0)
    time.sleep(25)
print(f"DGX PUSH TRIP TIMEOUT claimed={claimed} (check repo state via API)", flush=True)
sys.exit(1)
