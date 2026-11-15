"""H28 K4 probe snapshot loop v2 (runs on this VM).

Transport fix vs v1: gzip+base64 through the Actions log corrupted
(Incorrect padding). v2 pulls the raw jsonl lines (they are plain JSON
text) plus the target-side md5 of each file; a snapshot is only
committed to sync/h28 when the reconstructed bytes md5-match the
target's own md5. A mismatch keeps the last good branch copy.

Every trip (target cadence 120s; if a trip takes longer, the next
starts immediately -- shortest reachable interval, never 20 min):
  dispatch a pull script to L20 via cloud-assistant; L20 -> ssh DGX ->
  ssh spot-28; target prints chain status, counts, summaries, md5s and
  the raw per-game jsonl contents between markers.
Exits: DONE (both tags 200 + CHAIN_ALL_DONE), MACHINE_GONE (target ssh
fails on 2 consecutive trips), or DEADLINE.
"""
import base64, hashlib, json, os, re, sys, time

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
SNAP = BASE + "/exam_snapshots/h28"
os.makedirs(SNAP, exist_ok=True)
LOGF = SNAP + "/snaploop.log"
OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"
L20 = "i-uf62k0o6zbe6j7zxnzg5"
TAGS = ["k4v2r1_e0", "k4v2g1_e0"]
INTERVAL = 120
DEADLINE = time.time() + 120 * 60
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+")

seen_runs = set()
branch_counts = {}

def log_line(s):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
    print(line, flush=True)
    with open(LOGF, "a") as f:
        f.write(line + "\n")

def strip_ts(ln):
    return TS_RE.sub("", ln)

def pull_script(marker):
    return f"""MARK={marker}
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK={marker}
TGT=47.116.17.244
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
echo "$MARK DGX-BEGIN"
timeout -k 30 180 ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT" 'bash -s' <<'TGTEOF'
echo "TGT-BEGIN"
date '+%F %T %Z'
echo "--- CHAIN STATUS ---"
cat /root/carbon/data/k4probe.status 2>/dev/null || echo NO-STATUS
echo "--- COUNTS ---"
wc -l /root/carbon/data/bigeval_live/k4v2r1_e0.jsonl /root/carbon/data/bigeval_live/k4v2g1_e0.jsonl 2>/dev/null || echo NO-LIVE-FILES
echo "bigeval-procs: $(pgrep -fc bigeval.py || echo 0)"
cat /proc/loadavg
echo "--- SUMMARIES ---"
grep -h '"tag": "k4v2' /root/carbon/data/bigeval.jsonl 2>/dev/null || echo NO-SUMMARIES-YET
echo "--- MD5 ---"
md5sum /root/carbon/data/bigeval_live/k4v2r1_e0.jsonl /root/carbon/data/bigeval_live/k4v2g1_e0.jsonl 2>/dev/null || echo NO-MD5
echo "--- RAW k4v2r1_e0 ---"
cat /root/carbon/data/bigeval_live/k4v2r1_e0.jsonl 2>/dev/null || echo NOFILE
echo "--- RAW k4v2g1_e0 ---"
cat /root/carbon/data/bigeval_live/k4v2g1_e0.jsonl 2>/dev/null || echo NOFILE
echo "TGT-END"
TGTEOF
echo "TGT-RC=$?"
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
"""

def dispatch(script):
    gh_api.dispatch(OWNER, REPO, WF, ref="main",
                    inputs={"instance_id": L20, "region": "cn-shanghai",
                            "command_b64": base64.b64encode(script.encode()).decode()})

def claim(marker, timeout_s=420):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(15)
        try:
            r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30")
        except Exception as e:
            log_line(f"claim list error: {e}")
            continue
        for run in r.get("workflow_runs", [])[:30]:
            rid = run["id"]
            if rid in seen_runs or run["status"] != "completed":
                continue
            seen_runs.add(rid)
            try:
                jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs")
            except Exception:
                continue
            for j in jobs.get("jobs", []):
                try:
                    log = gh_api.get_job_logs(OWNER, REPO, j["id"])
                except Exception:
                    continue
                if isinstance(log, str) and marker in log:
                    return log
    return None

def segment(log, start_key, stop_keys):
    i = log.find(start_key)
    if i < 0:
        return None
    seg = log[i + len(start_key):]
    end = len(seg)
    for stop in stop_keys:
        j = seg.find(stop)
        if 0 <= j < end:
            end = j
    return seg[:end]

def target_md5s(log):
    seg = segment(log, "--- MD5 ---", ["--- RAW"])
    out = {}
    if seg is None:
        return out
    for ln in seg.splitlines():
        ln = strip_ts(ln).strip()
        m = re.match(r"^([0-9a-f]{32})\s+.*/(k4v2\w+)\.jsonl$", ln)
        if m:
            out[m.group(2)] = m.group(1)
    return out

def extract_raw(log, tag):
    seg = segment(log, f"--- RAW {tag} ---", ["--- RAW", "TGT-END"])
    if seg is None:
        return None
    lines = []
    for ln in seg.splitlines():
        ln = strip_ts(ln).rstrip("\r")
        if ln.startswith("{"):
            lines.append(ln)
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode()

def put_branch(path, data_bytes, msg):
    sha = None
    try:
        r = gh_api.get_file(OWNER, REPO, path, ref="sync/h28")
        if isinstance(r, dict):
            sha = r.get("sha")
    except Exception:
        sha = None
    gh_api.put_file(OWNER, REPO, path, data_bytes, msg, sha=sha, branch="sync/h28")
    b = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/sync%2Fh28")
    return b["commit"]["sha"][:8], b["commit"]["commit"]["author"]["date"]

def main():
    log_line("SNAPLOOP_V2_START h28 K4 probe")
    trip = 0
    gone_streak = 0
    while time.time() < DEADLINE:
        trip += 1
        t0 = time.time()
        marker = "S28K4SNP" + time.strftime("%H%M%S")
        try:
            dispatch(pull_script(marker))
        except Exception as e:
            log_line(f"trip={trip} dispatch error: {e}")
            time.sleep(30)
            continue
        log = claim(marker)
        if log is None:
            log_line(f"trip={trip} NOT-CLAIMED marker={marker}")
            time.sleep(20)
            continue
        target_ok = "TGT-BEGIN" in log and "TGT-END" in log
        if not target_ok:
            gone_streak += 1
            log_line(f"trip={trip} TARGET-UNREACHABLE streak={gone_streak} counts={branch_counts}")
            if gone_streak >= 2:
                log_line(f"trip={trip} MACHINE_GONE final_counts={branch_counts}")
                return
        else:
            gone_streak = 0
            status_lines = [strip_ts(ln) for ln in log.splitlines() if "CHAIN_" in ln]
            done_all = any("CHAIN_ALL_DONE" in s for s in status_lines)
            summ_present = [t for t in TAGS if f'"tag": "{t}"' in log]
            md5s = target_md5s(log)
            for tag in TAGS:
                if tag not in md5s:
                    log_line(f"trip={trip} {tag} NO-TARGET-MD5 (file absent?) branch={branch_counts.get(tag)}")
                    continue
                data = extract_raw(log, tag)
                if data is None:
                    log_line(f"trip={trip} {tag} NO-RAW-SEGMENT")
                    continue
                mine = hashlib.md5(data).hexdigest()
                if mine != md5s[tag]:
                    log_line(f"trip={trip} {tag} MD5-MISMATCH mine={mine[:8]} tgt={md5s[tag][:8]} lines={data.count(bytes([10]))} -- keeping last good branch copy")
                    continue
                n = data.count(b"\n")
                open(os.path.join(SNAP, tag + ".jsonl"), "wb").write(data)
                if branch_counts.get(tag) != n:
                    sha8, cdate = put_branch(f"synced/h28/bigeval_live/{tag}.jsonl", data,
                                             f"h28 snap {tag} {n} lines")
                    branch_counts[tag] = n
                    log_line(f"trip={trip} {tag}={n} md5ok -> sync/h28 @{sha8} {cdate} status={status_lines[-1] if status_lines else '?'} summ={summ_present}")
                else:
                    log_line(f"trip={trip} {tag}={n} md5ok unchanged status={status_lines[-1] if status_lines else '?'} summ={summ_present}")
            summ_lines = []
            for ln in log.splitlines():
                ln2 = strip_ts(ln)
                if '"tag": "k4v2' in ln2 and '"mean"' in ln2:
                    i = ln2.find("{")
                    if i >= 0:
                        try:
                            rec = json.loads(ln2[i:])
                            if rec.get("tag") in TAGS:
                                summ_lines.append(json.dumps(rec, ensure_ascii=False))
                        except Exception:
                            pass
            if summ_lines:
                payload = ("\n".join(sorted(set(summ_lines))) + "\n").encode()
                sha8, cdate = put_branch("synced/h28/bigeval.jsonl", payload,
                                         "h28 snap summaries")
                log_line(f"trip={trip} summaries={len(summ_lines)} -> sync/h28 @{sha8}")
            if all(branch_counts.get(t) == 200 for t in TAGS) and done_all:
                log_line(f"trip={trip} ALL_DONE counts={branch_counts}")
                return
        dt = time.time() - t0
        if dt < INTERVAL:
            time.sleep(INTERVAL - dt)
    log_line(f"DEADLINE counts={branch_counts}")

if __name__ == "__main__":
    main()
