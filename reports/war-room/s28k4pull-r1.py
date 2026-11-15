"""H28 K4 probe r1 pull: 5-line chunks (spot-26 proven method).

Reads k4v2r1_e0.jsonl off spot-28 via L20 -> DGX -> spot-28 ssh, 5 lines
per trip (sed -n range), claims each Actions run by its unique marker,
verifies every line (JSON parse + (seed, side, margin) triple present in
the branch summary's games table), appends to the local accumulation
file, and put_files the accumulation to sync/h28 after each batch.
No whole-file cat, no md5 gating on the transport (per-line JSON +
summary-table verification replaces it). Never touches the exam or the
on-box snapshot loop.
"""
import base64, json, os, re, sys, time

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
SNAP = BASE + "/exam_snapshots/h28"
os.makedirs(SNAP, exist_ok=True)
LOGF = SNAP + "/pull-r1.log"
OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"
L20 = "i-uf62k0o6zbe6j7zxnzg5"
TAG = "k4v2r1_e0"
TOTAL = 200
CHUNK = 5
BATCH = 4
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+")

seen_runs = set()

def log_line(s):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
    print(line, flush=True)
    with open(LOGF, "a") as f:
        f.write(line + "\n")

def summary_games():
    # 2026-11-13 ~00:15: branch bigeval.jsonl is JSONL (smoke + r1 + g1
    # merged by be6fa25c); parse per line and pick our tag.
    r = gh_api.get_file(OWNER, REPO, "synced/h28/bigeval.jsonl", ref="sync/h28")
    txt = base64.b64decode(r["content"]).decode()
    rec = None
    for ln in txt.splitlines():
        if not ln.strip():
            continue
        d = json.loads(ln)
        if d.get("tag") == TAG:
            rec = d
            break
    assert rec is not None, f"tag {TAG} not in branch bigeval.jsonl"
    games = {}
    for seed, side, margin in rec["games"]:
        games[(seed, side, margin)] = True
    return games, rec

def pull_script(marker, start, end):
    return f"""MARK={marker}
echo "$MARK L20-BEGIN"
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK={marker}
TGT=47.116.17.244
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
timeout -k 30 180 ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT" 'bash -s' <<'TGTEOF'
echo "TGT-BEGIN"
wc -l /root/carbon/data/bigeval_live/{TAG}.jsonl
echo "--- CHUNK {start}-{end} ---"
sed -n '{start},{end}p' /root/carbon/data/bigeval_live/{TAG}.jsonl
echo "--- CHUNK-END ---"
echo "TGT-END"
TGTEOF
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END"
"""

def dispatch(script):
    gh_api.dispatch(OWNER, REPO, WF, ref="main",
                    inputs={"instance_id": L20, "region": "cn-shanghai",
                            "command_b64": base64.b64encode(script.encode()).decode()})

def fetch_log(job_id):
    # 2026-11-13 ~00:10: urllib blob reads die with IncompleteRead /
    # RemoteDisconnected through the egress proxy; curl --http1.1 works.
    # fetch_log_curl.fetch writes the log to a temp file; read it back.
    import tempfile, os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from fetch_log_curl import fetch as curl_fetch
    for _ in range(3):
        tmp = tempfile.mktemp(prefix="pull-log-", suffix=".log")
        try:
            if curl_fetch(job_id, tmp):
                with open(tmp, errors="replace") as f:
                    return f.read()
        except Exception as e:
            log_line(f"log fetch retry job={job_id} err={str(e)[:80]}")
        finally:
            try:
                _os.unlink(tmp)
            except OSError:
                pass
        time.sleep(5)
    return None

def claim(marker, timeout_s=420, since=None):
    # 2026-11-13 owner rule: shared 5000/hr API pool -- poll every 60s and
    # only inspect runs created after `since` (batch dispatch time).
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(60)
        try:
            r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30")
        except Exception as e:
            log_line(f"claim list error: {e}")
            continue
        for run in r.get("workflow_runs", [])[:30]:
            rid = run["id"]
            if rid in seen_runs or run["status"] != "completed":
                continue
            if since and run.get("created_at", "") < since:
                continue
            try:
                jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs")
            except Exception:
                continue
            log_ok = True
            for j in jobs.get("jobs", []):
                log = fetch_log(j["id"])
                if log is None:
                    log_ok = False
                    continue
                if isinstance(log, str) and marker in log:
                    seen_runs.add(rid)
                    return log
            if log_ok:
                seen_runs.add(rid)
    return None

def extract_lines(log, start, end):
    key = f"--- CHUNK {start}-{end} ---"
    i = log.find(key)
    if i < 0:
        return None
    seg = log[i + len(key):]
    j = seg.find("--- CHUNK-END ---")
    if j >= 0:
        seg = seg[:j]
    lines = []
    for ln in seg.splitlines():
        ln = TS_RE.sub("", ln).rstrip("\r")
        if ln.startswith("{"):
            lines.append(ln)
    return lines

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
    games, rec = summary_games()
    log_line(f"PULL_START r1 summary n={rec['n']} wins={rec['wins']} mean={rec['mean']} games_table={len(games)}")
    acc_path = os.path.join(SNAP, TAG + ".jsonl")
    # buffer: chunk (a, b) -> verified lines, kept in file order on commit
    buf = {}
    if os.path.exists(acc_path):
        old = [l for l in open(acc_path).read().splitlines() if l.strip()]
        for i in range(0, len(old), CHUNK):
            buf[(i + 1, min(i + CHUNK, TOTAL))] = old[i:i + CHUNK]
        log_line(f"resume: local accumulation already {len(old)} lines in {len(buf)} chunks")
    chunks = [(s + 1, min(s + CHUNK, TOTAL)) for s in range(0, TOTAL, CHUNK)]
    pending = [c for c in chunks if c not in buf]
    log_line(f"chunks pending: {len(pending)} of {len(chunks)}")
    for bi in range(0, len(pending), BATCH):
        batch = pending[bi:bi + BATCH]
        markers = {}
        batch_since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for (a, b) in batch:
            m = f"S28K4PL{time.strftime('%H%M%S')}{a:03d}"
            markers[m] = (a, b)
            dispatch(pull_script(m, a, b))
            time.sleep(2)
        for m, (a, b) in markers.items():
            log = claim(m, since=batch_since)
            if log is None:
                log_line(f"chunk {a}-{b} NOT-CLAIMED marker={m} -- will retry next round")
                continue
            lines = extract_lines(log, a, b)
            if lines is None:
                log_line(f"chunk {a}-{b} NO-CHUNK-SEGMENT -- will retry")
                continue
            ok = True
            for ln in lines:
                try:
                    r = json.loads(ln)
                except Exception as e:
                    log_line(f"chunk {a}-{b} BAD-JSON {ln[:80]} err={e}")
                    ok = False
                    break
                if (r["seed"], r["side"], r["margin"]) not in games:
                    log_line(f"chunk {a}-{b} TRIPLE-NOT-IN-SUMMARY seed={r['seed']}")
                    ok = False
                    break
            if not ok:
                log_line(f"chunk {a}-{b} FAILED verification -- will retry")
                continue
            expect = b - a + 1
            if len(lines) != expect:
                log_line(f"chunk {a}-{b} SHORT got {len(lines)} expect {expect} -- retrying")
                continue
            buf[(a, b)] = lines
            log_line(f"chunk {a}-{b} OK lines={len(lines)}")
        # assemble contiguous prefix from line 1 and commit it
        acc = []
        for (a, b) in chunks:
            if (a, b) in buf:
                acc.extend(buf[(a, b)])
            else:
                break
        if acc:
            data = ("\n".join(acc) + "\n").encode()
            seeds = [json.loads(l)["seed"] for l in acc]
            sha8, cdate = put_branch(f"synced/h28/bigeval_live/{TAG}.jsonl", data,
                                     f"h28 r1 pull {len(acc)}/200 lines seeds {min(seeds)}-{max(seeds)}")
            with open(acc_path, "w") as f:
                f.write("\n".join(acc) + "\n")
            log_line(f"committed {len(acc)}/200 contiguous -> sync/h28 @{sha8} {cdate}")
        if len(acc) >= TOTAL:
            break
    acc_seeds = {json.loads(l)["seed"] for l in sum(buf.values(), [])}
    missing = sorted(s for (s, _, _) in games if s not in acc_seeds)
    total = sum(len(v) for v in buf.values())
    log_line(f"after chunks: buffered={total} missing_seeds={missing}")
    if total == TOTAL and not missing:
        log_line("PULL_DONE r1 200/200 seeds 997000-997199 verified against summary table")
    else:
        log_line("PULL_INCOMPLETE -- rerun to retry failed chunks / gap-fill by seed grep")

if __name__ == "__main__":
    main()
