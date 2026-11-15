import base64, datetime, sys, time
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
MARKER = "E3PUSH1" + time.strftime("%H%M%S")
INSTANCE = "i-uf62k0o6zbe6j7zxnzg5"  # gpu-l20-5
DGXK = "-i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
STUK = "-i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112"
EXP_MD5 = "7b7620499411a088cf4dc17c0e929bd7"
EXP_SIZE = "4576790"

# Stage A runs on L20. Stage B (Studio) is a heredoc leg at the very end.
script = f"""set +e
echo {MARKER}-BEGIN
hostname; date
EXP_MD5={EXP_MD5}; EXP_SIZE={EXP_SIZE}
RESULT="NONE"
echo '=== A: L20 local pack + auth ==='
PACK=$(find /root/carbon -maxdepth 3 -name 'echo3_int8.py' 2>/dev/null | head -1)
if [ -n "$PACK" ]; then echo "PACK=$PACK $(stat -c %s "$PACK") $(md5sum "$PACK" | cut -d' ' -f1)"; else echo "PACK=NONE-ON-L20"; fi
AUTH_OK=0
timeout -k 5 20 ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new -T git@github.com </dev/null 2>&1 | head -1 | grep -q "successfully authenticated" && AUTH_OK=1
if [ $AUTH_OK -eq 0 ] && command -v gh >/dev/null; then gh auth status >/dev/null 2>&1 && AUTH_OK=1; fi
echo "L20-AUTH-OK=$AUTH_OK"
if [ -n "$PACK" ] && [ "$(md5sum "$PACK" | cut -d' ' -f1)" = "$EXP_MD5" ] && [ "$(stat -c %s "$PACK")" = "$EXP_SIZE" ] && [ $AUTH_OK -eq 1 ]; then
  echo '=== A: push from L20 ==='
  W=$(mktemp -d /tmp/e3push.XXXXXX); cd "$W" || exit 0
  export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new"
  timeout 240 git clone --depth 1 --filter=blob:none --sparse git@github.com:robertyang87/carbon-league.git repo 2>&1 | tail -2
  cd repo 2>/dev/null && timeout 120 git sparse-checkout set code/packs 2>&1 | tail -1
  if [ -d code/packs ]; then
    cp "$PACK" code/packs/echo3_int8.py
    git config user.name "carbon-ci"; git config user.email "ci@users.noreply.github.com"
    git add code/packs/echo3_int8.py
    if git status --porcelain | grep -q echo3_int8; then
      git commit -m "data: add Echo3 int8 pack (m3s2r1, md5 7b762049) for ladder submission" 2>&1 | tail -1
      BLOB=$(git rev-parse HEAD:code/packs/echo3_int8.py); BSZ=$(git cat-file -s "$BLOB"); echo "BLOB=$BLOB SIZE=$BSZ"
      timeout 300 git push origin HEAD:main 2>&1 | tail -3
      if git ls-remote origin refs/heads/main | grep -q "$(git rev-parse HEAD)"; then RESULT="A-L20-PUSHED-MAIN $(git rev-parse HEAD) blob=$BLOB size=$BSZ";
      else timeout 300 git push origin HEAD:refs/heads/echo3-pack 2>&1 | tail -2; RESULT="A-L20-PUSHED-BRANCH-ECH03PACK $(git rev-parse HEAD) blob=$BLOB size=$BSZ"; fi
    else RESULT="A-ALREADY-IN-MAIN"; fi
  else echo "A-CLONE-FAILED"; fi
  cd /; rm -rf "$W"
fi
echo "RESULT-A=$RESULT"
if [ "$RESULT" = "NONE" ]; then
echo '=== B: Studio leg ==='
timeout -k 30 600 ssh {STUK} arm@10.211.43.68 'bash -s' <<'STEOF'
set +e
MARK=__MARKER__
EXP_MD5=__EXP_MD5__; EXP_SIZE=__EXP_SIZE__
echo "STUDIO-BEGIN $(hostname)"
F=/tmp/echo3_int8.py
if [ ! -f "$F" ]; then echo "STUDIO-PACK-MISSING"; echo "$MARK RESULT=NONE-B"; exit 0; fi
M=$(md5 -q "$F"); S=$(stat -f %z "$F"); echo "STUDIO-PACK md5=$M size=$S"
if [ "$M" != "$EXP_MD5" ] || [ "$S" != "$EXP_SIZE" ]; then echo "STUDIO-PACK-MISMATCH"; echo "$MARK RESULT=NONE-B"; exit 0; fi
command -v git >/dev/null || {{ echo "STUDIO-NO-GIT"; echo "$MARK RESULT=NONE-B"; exit 0; }}
timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new -T git@github.com </dev/null 2>&1 | head -1
W=$(mktemp -d /tmp/e3push.XXXXXX); cd "$W" || exit 0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new"
timeout 240 git clone --depth 1 --filter=blob:none --sparse git@github.com:robertyang87/carbon-league.git repo 2>&1 | tail -2
cd repo 2>/dev/null && timeout 120 git sparse-checkout set code/packs 2>&1 | tail -1
if [ -d code/packs ]; then
  cp "$F" code/packs/echo3_int8.py
  git config user.name "carbon-ci"; git config user.email "ci@users.noreply.github.com"
  git add code/packs/echo3_int8.py
  if git status --porcelain | grep -q echo3_int8; then
    git commit -m "data: add Echo3 int8 pack (m3s2r1, md5 7b762049) for ladder submission" 2>&1 | tail -1
    BLOB=$(git rev-parse HEAD:code/packs/echo3_int8.py); BSZ=$(git cat-file -s "$BLOB"); echo "BLOB=$BLOB SIZE=$BSZ"
    timeout 300 git push origin HEAD:main 2>&1 | tail -3
    if git ls-remote origin refs/heads/main | grep -q "$(git rev-parse HEAD)"; then echo "$MARK RESULT=B-STUDIO-PUSHED-MAIN $(git rev-parse HEAD) blob=$BLOB size=$BSZ";
    else timeout 300 git push origin HEAD:refs/heads/echo3-pack 2>&1 | tail -2; echo "$MARK RESULT=B-STUDIO-PUSHED-BRANCH $(git rev-parse HEAD) blob=$BLOB size=$BSZ"; fi
  else echo "$MARK RESULT=B-ALREADY-IN-MAIN"; fi
else echo "STUDIO-CLONE-FAILED"; echo "$MARK RESULT=NONE-B"; fi
cd /; rm -rf "$W"
STEOF
echo "B-LEG-RC=$?"
fi
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
                seg = log[max(0, i - 200):(j + len(MARKER) + 4 if j >= 0 else i + 9000)]
                print(seg, flush=True)
                with open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/echo3-push1-claim.log", "w") as f:
                    f.write(log)
                print("=== PUSH TRIP DONE ===", flush=True)
                sys.exit(0)
    time.sleep(25)
print("PUSH TRIP CLAIM TIMEOUT (effect may still have landed - verify via gh_api)", flush=True)
sys.exit(1)
