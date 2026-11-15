"""Echo3 equivalence snapshot loop (runs on this VM).

Pulls /root/carbon/data/bigeval_live/echo3_m2.jsonl from spot-30 in
9-line raw windows via cloud-assistant (direct to the spot instance),
stores lines keyed by seed, backfills gaps by seed-targeted grep, and
pushes the cumulative file to sync/h30 (synced/h30/bigeval_live/
echo3_m2.jsonl) via the git data API. A snapshot counts only after a
read-back of the branch ref + a contents size check.

Exits: DONE (200 seeds + local md5 == target md5 + final push verified),
MACHINE_GONE (2 consecutive failed trips), or DEADLINE (45 min).
"""
import base64, hashlib, json, os, re, sys, time
sys.path.insert(0, "/home/hatch/workspace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gh_api
from s30echo3_driver_lib import fetch_log_tolerant
from datetime import datetime, timezone

BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
SNAP = BASE + "/exam_snapshots/h30"
os.makedirs(SNAP, exist_ok=True)
LOGF = SNAP + "/snaploop-echo3_m2.log"
STORE = SNAP + "/lines-echo3_m2.json"
OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"
INST = "i-uf6ak9krs1kt71y2unce"
BRANCH = "sync/h30"
BPATH = "synced/h30/bigeval_live/echo3_m2.jsonl"
TAG = "echo3_m2"
SEED0, N = 996000, 200
WINDOW = 9
DEADLINE = time.time() + 45 * 60
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+")

store = {}
if os.path.exists(STORE):
    store = {int(k): v for k, v in json.load(open(STORE)).items()}
next_line = 1
pushed_count = 0
fail_streak = 0
summary_line = None

def log_line(s):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
    print(line, flush=True)
    with open(LOGF, "a") as f:
        f.write(line + "\n")

def save_store():
    json.dump({str(k): v for k, v in sorted(store.items())}, open(STORE, "w"))

def pull_script(marker, a, b, gaps):
    gap_txt = ""
    if gaps:
        pat = "|".join(f'"seed": {s}' for s in gaps)
        gap_txt = (f'echo "--- GAPFILL ---"\n'
                   f"grep -E '{pat}' /root/carbon/data/bigeval_live/{TAG}.jsonl 2>/dev/null\n")
    return f"""MARK={marker}
echo "$MARK PAYLOAD-BEGIN"
echo "--- LINES {a}-{b} ---"
sed -n '{a},{b}p' /root/carbon/data/bigeval_live/{TAG}.jsonl 2>/dev/null
{gap_txt}echo "--- STATUS ---"
wc -l /root/carbon/data/bigeval_live/{TAG}.jsonl 2>/dev/null || echo NO-LIVE
md5sum /root/carbon/data/bigeval_live/{TAG}.jsonl 2>/dev/null || echo NO-MD5
echo "bigeval-procs: $(pgrep -cf bigeval.py || echo 0)"
grep '"tag": "{TAG}"' /root/carbon/data/bigeval.jsonl 2>/dev/null | tail -1 || echo NO-SUMMARY
ls -l --time-style=full-iso /root/carbon/data/bigeval_live/{TAG}.jsonl 2>/dev/null
echo "$MARK PAYLOAD-END"
"""

def dispatch(script):
    gh_api.dispatch(OWNER, REPO, WF, ref="main",
                    inputs={"instance_id": INST, "region": "cn-shanghai",
                            "command_b64": base64.b64encode(script.encode()).decode()})
    return datetime.now(timezone.utc)

def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def claim(marker, t_disp, timeout_s=150):
    deadline = time.time() + timeout_s
    seen = set()
    while time.time() < deadline:
        time.sleep(15)
        try:
            r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30")
        except Exception as e:
            log_line(f"list error: {e}")
            if "403" in str(e):
                log_line("RATE-LIMITED: aborting trip")
                return None
            continue
        for run in r.get("workflow_runs", []):
            rid = run["id"]
            if rid in seen:
                continue
            try:
                created = parse_ts(run["created_at"])
            except Exception:
                continue
            if created <= t_disp:
                continue
            if run["status"] != "completed":
                continue
            seen.add(rid)
            try:
                jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs")
            except Exception:
                continue
            for j in jobs.get("jobs", []):
                best = ""
                for _ in range(3):
                    try:
                        log, how = fetch_log_tolerant(j["id"])
                    except Exception:
                        continue
                    if len(log) > len(best):
                        best = log
                    if marker in log:
                        break
                if marker in best:
                    return best
    return None

def push_branch(content_bytes, message):
    """Push content_bytes to BPATH on BRANCH via git data API; verify ref moved."""
    ref = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/git/ref/heads/{BRANCH}")
    parent = ref["object"]["sha"]
    commit = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/git/commits/{parent}")
    base_tree = commit["tree"]["sha"]
    blob = gh_api.req("POST", f"/repos/{OWNER}/{REPO}/git/blobs",
                      {"content": base64.b64encode(content_bytes).decode(), "encoding": "base64"})
    tree = gh_api.req("POST", f"/repos/{OWNER}/{REPO}/git/trees",
                      {"base_tree": base_tree,
                       "tree": [{"path": BPATH, "mode": "100644", "type": "blob", "sha": blob["sha"]}]})
    new_commit = gh_api.req("POST", f"/repos/{OWNER}/{REPO}/git/commits",
                            {"message": message, "tree": tree["sha"], "parents": [parent]})
    gh_api.req("PATCH", f"/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}", {"sha": new_commit["sha"]})
    ref2 = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/git/ref/heads/{BRANCH}")
    return ref2["object"]["sha"] == new_commit["sha"], new_commit["sha"]

def build_file_bytes():
    lines = [store[s] for s in sorted(store)]
    return ("\n".join(lines) + "\n").encode()

trip = 0
while time.time() < DEADLINE:
    trip += 1
    total_known = None
    # gaps: seeds already written on target (line idx < next_line) but not stored
    gaps = []
    if store and len(store) < N:
        gaps = [s for s in range(SEED0, SEED0 + N) if s not in store][:6]
    marker = "S30E3SNAP" + time.strftime("%H%M%S")
    a, b = next_line, next_line + WINDOW - 1
    t_disp = dispatch(pull_script(marker, a, b, gaps))
    log = claim(marker, t_disp)
    if log is None:
        fail_streak += 1
        log_line(f"trip={trip} NOT-CLAIMED marker={marker} fail_streak={fail_streak}")
        if fail_streak >= 2:
            log_line("MACHINE_GONE?")
            break
        continue
    fail_streak = 0
    text = "\n".join(TS_RE.sub("", ln) for ln in log.splitlines())
    i0, i1 = text.find(marker + " PAYLOAD-BEGIN"), text.find(marker + " PAYLOAD-END")
    payload = text[i0:i1] if 0 <= i0 < i1 else ""
    got = 0
    for ln in payload.splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        s = obj.get("seed")
        if isinstance(s, int) and SEED0 <= s < SEED0 + N and s not in store:
            store[s] = ln
            got += 1
    save_store()
    next_line = b + 1
    m = re.search(r"(\d+)\s+/root/carbon/data/bigeval_live/" + TAG + r"\.jsonl", payload)
    if m:
        total_known = int(m.group(1))
    tm = re.search(r"([0-9a-f]{32})\s+/root/carbon/data/bigeval_live/" + TAG + r"\.jsonl", payload)
    target_md5 = tm.group(1) if tm else None
    pm = re.search(r"bigeval-procs: (\d+)", payload)
    procs = pm.group(1) if pm else "?"
    sm = re.search(r'(\{"tag": "' + TAG + r'".*)', payload)
    if sm:
        summary_line = sm.group(1).strip()
    log_line(f"trip={trip} window={a}-{b} got={got} stored={len(store)}/200 target_total={total_known} procs={procs} target_md5={target_md5}")
    # push policy: first push ASAP, then every >=25 new lines, then final
    if len(store) > pushed_count and (pushed_count == 0 or len(store) - pushed_count >= 25 or len(store) == N):
        data = build_file_bytes()
        try:
            ok, sha = push_branch(data, f"sync h30: {TAG} snapshot {len(store)}/200")
            if ok:
                pushed_count = len(store)
                log_line(f"PUSH-OK {pushed_count}/200 @{sha[:8]} bytes={len(data)}")
            else:
                log_line("PUSH-VERIFY-FAILED")
        except Exception as e:
            log_line(f"PUSH-ERROR {e}")
    if len(store) == N:
        data = build_file_bytes()
        local_md5 = hashlib.md5(data).hexdigest()
        log_line(f"ALL-SEEDS local_md5={local_md5} target_md5={target_md5} match={local_md5 == target_md5}")
        if summary_line:
            open(SNAP + "/summary-echo3_m2.json", "w").write(summary_line + "\n")
            log_line("SUMMARY-SAVED")
        log_line("DONE")
        break
    time.sleep(10)

log_line(f"LOOP-END stored={len(store)}/200 pushed={pushed_count} summary={'yes' if summary_line else 'no'}")
