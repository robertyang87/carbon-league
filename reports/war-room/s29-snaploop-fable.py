"""C1 Fable r1 snapshot loop v3 (runs on this VM).

War-room order 23:38 (supersedes the b64 approach): per-game jsonl lines
are pulled RAW in small windows (no gzip/b64 whole-file blobs - the
Actions-log pipeline corrupts/truncates them), and summaries go to
PER-TAG files (synced/h29/summaries/<tag>.json) - two loops must never
co-write one shared file.

- Each trip pulls a 50-line window per tag (sed -n 'A,Bp'), ~10KB raw
  text, well inside the ~39KB log payload proven to survive intact.
- Lines are keyed by their "seed" value; the store persists across
  trips/restarts (chunks-fable.json). A file is complete when its 200
  seeds (seed0..seed0+199) are all present.
- Gaps are backfilled by seed-targeted grep -E (war-room prescribed).
- Claim re-fetches each job log up to 3x and keeps the LONGEST copy
  (blob downloads have been seen to truncate mid-stream).
Exits: ALL_DONE (all tags complete + CHAIN_ALL_DONE), MACHINE_GONE
(target unreachable on 2 consecutive trips), or DEADLINE.
"""
import base64, json, os, sys, time

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
SNAP = BASE + "/exam_snapshots/h29"
os.makedirs(SNAP, exist_ok=True)
LOGF = SNAP + "/snaploop-fable.log"
STORE = SNAP + "/lines-fable.json"
LINEPOS = SNAP + "/linepos-fable.json"
OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"
L20 = "i-uf62k0o6zbe6j7zxnzg5"
BRANCH = "sync/h29"
TAGS = ["fabler1_m2", "fabler1_echo2"]
STATUS = "fabler1_cal.status"
WINDOW = 9
INTERVAL = 120
DEADLINE = 1794503323 + 720 * 60  # v3.5 restart 01:08:43 CST + 720 min (war-room ruling 01:26); absolute end ~13:08 CST; do not extend without a new ruling

seen_runs = set()
branch_counts = {}

def log_line(s):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
    print(line, flush=True)
    with open(LOGF, "a") as f:
        f.write(line + "\n")

def pull_script(marker, windows, gaps, focus):
    # ONE focus tag per trip, LINES section FIRST (right after TGT-BEGIN):
    # the log-download channel truncates fetches at ~15KB, so the payload
    # that must survive is the 50-line window (~10KB) + tiny status lines.
    # Summaries/counts come after and may be cut; they are re-pulled
    # every trip anyway.
    a, b = windows[focus]
    lines_sections = [
        f'echo "--- LINES {focus} {a}-{b} ---"\n'
        f"sed -n '{a},{b}p' /root/carbon/data/bigeval_live/{focus}.jsonl 2>/dev/null"]
    if gaps.get(focus):
        pat = "|".join(f'"seed": {s}' for s in gaps[focus])
        lines_sections.append(
            f'echo "--- GAPFILL {focus} ---"\n'
            f'grep -E \'{pat}\' /root/carbon/data/bigeval_live/{focus}.jsonl 2>/dev/null')
    lines_txt = "\n".join(lines_sections)
    script = """MARK=__MARKER__
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK=__MARKER__
TGT=47.117.170.99
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
echo "$MARK DGX-BEGIN"
timeout -k 30 180 ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT" 'bash -s' <<'TGTEOF'
echo "TGT-BEGIN"
date '+%F %T %Z'
__LINES__
echo "--- CHAIN STATUS ---"
cat /root/carbon/data/__STATUS__ 2>/dev/null || echo NO-STATUS
echo "--- COUNTS ---"
wc -l __WCFILES__ 2>/dev/null || echo NO-LIVE-FILES
echo "bigeval-procs: $(pgrep -fc bigeval.py || echo 0)"
cat /proc/loadavg
echo "--- SUMMARIES ---"
/root/carbon/.venv/bin/python -c "
import json
try:
    for ln in open('/root/carbon/data/bigeval.jsonl'):
        s = json.loads(ln)
        if str(s.get('tag','')).startswith('fabler1'):
            s.pop('games', None)
            print(json.dumps(s, ensure_ascii=False))
except FileNotFoundError:
    print('NO-SUMMARIES-YET')
"
echo "TGT-END"
TGTEOF
echo "TGT-RC=$?"
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
"""
    wc = " ".join(f"/root/carbon/data/bigeval_live/{t}.jsonl" for t in TAGS)
    return (script.replace("__MARKER__", marker)
                 .replace("__STATUS__", STATUS)
                 .replace("__WCFILES__", wc)
                 .replace("__LINES__", lines_txt))

def dispatch(script):
    gh_api.dispatch(OWNER, REPO, WF, ref="main",
                    inputs={"instance_id": L20, "region": "cn-shanghai",
                            "command_b64": base64.b64encode(script.encode()).decode()})

def _blob_loc(job_id):
    """Resolve a job's log blob URL (302 -> Azure; no auth on the blob).
    Returns (loc, direct_text); direct_text set when there is no redirect."""
    import urllib.error, urllib.request
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    url = f"{gh_api.API}/repos/{OWNER}/{REPO}/actions/jobs/{job_id}/logs"
    r = urllib.request.Request(url, method="GET", headers={"User-Agent": "muse-agent"})
    gh_api.add_surrogate_to_request(r, "custom.github", allowed_hosts=["api.github.com"])
    try:
        with opener.open(r, timeout=30) as resp:
            return None, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location")
        if not loc:
            raise
        return loc, None


def _get_plain(loc):
    """Plain blob GET -> (text, complete). On IncompleteRead the partial
    prefix (~15364 bytes on this egress) is returned, not thrown away."""
    import urllib.request
    req = urllib.request.Request(loc, method="GET", headers={"User-Agent": "muse-agent"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read().decode("utf-8", errors="replace"), True
    except Exception as e:
        partial = getattr(e, "partial", None)
        if partial:
            if isinstance(partial, bytes):
                return partial.decode("utf-8", errors="replace"), False
            return str(partial), False
        raise


def _get_range(loc, start, end, attempts=6):
    """One bounded Range chunk. RemoteDisconnected fails fast, so a few
    attempts are cheap; measured 1-6 attempts per 6KB chunk tonight."""
    import urllib.request
    last = None
    for _attempt in range(attempts):
        try:
            req = urllib.request.Request(
                loc, method="GET",
                headers={"User-Agent": "muse-agent", "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=45) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001 - retry then surface
            last = e
            time.sleep(3)
    raise RuntimeError(f"range {start}-{end} failed: {last!r}")


def fetch_full(loc, probe_text):
    """Assemble the complete log in 6KB Range chunks (measured: a full
    41600-byte log assembled chunk by chunk). On stall, return the probe
    prefix - the seed store still banks partial progress per trip."""
    parts, start = [], 0
    try:
        while start <= 300000:
            data = _get_range(loc, start, start + 5999)
            parts.append(data)
            if len(data) < 6000:
                break
            start += 6000
    except Exception:
        pass  # prefix-tolerant: bank contiguous chunks assembled so far
    if parts:
        text = b"".join(parts).decode("utf-8", errors="replace")
        if len(text) > len(probe_text):
            return text
    return probe_text


def claim(marker, timeout_s=420, not_before=0):
    # API economy (owner rule 2026-11-13 00:36, shared quota pool):
    # 30s polls, and only runs created at/after this trip's dispatch
    # can be ours - older foreign runs are skipped without any
    # jobs/probe calls. created_at compares lexicographically (ISO Z).
    cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(not_before - 30)) if not_before else ""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        # quota throttle (war-room 2026-11-13 04:14): poll every 90s, not
        # 30s - the cloud run takes minutes, so detection stays prompt
        time.sleep(90)
        try:
            r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30")
        except Exception as e:
            log_line(f"claim list error: {e}")
            if "403" in str(e):
                time.sleep(60)
            continue
        for run in r.get("workflow_runs", [])[:30]:
            rid = run["id"]
            if rid in seen_runs or run["status"] != "completed":
                continue
            if cutoff and run.get("created_at", "") < cutoff:
                seen_runs.add(rid)
                continue
            seen_runs.add(rid)
            try:
                jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs")
            except Exception:
                continue
            for j in jobs.get("jobs", []):
                # Phase A: cheap probe. Measured 2026-11-13 00:33 on a
                # real snap log (41600 bytes): marker at byte 14088,
                # LINES payload starts 14320 - so the truncated plain
                # prefix (~15364) identifies our run, but the payload
                # itself needs the Range assembly of Phase B.
                try:
                    loc, direct = _blob_loc(j["id"])
                except Exception:
                    continue
                if direct is not None:
                    if marker in direct:
                        return direct
                    continue
                if not loc:
                    continue
                try:
                    probe, complete = _get_plain(loc)
                except Exception:
                    continue
                if complete:
                    if marker in probe:
                        return probe
                    continue
                if marker not in probe:
                    # marker may have drifted past the truncation point:
                    # one straddling chunk before giving up on this job
                    try:
                        chunk = _get_range(loc, 12000, 17999, attempts=2).decode("utf-8", errors="replace")
                    except Exception:
                        continue
                    if marker not in chunk:
                        continue
                # Phase B: full Range assembly for the payload
                return fetch_full(loc, probe)
    return None


def extract_lines(log, tag):
    """Collect raw jsonl lines for one tag from the claimed log.
    Returns {seed: line-text}. Lines are identified by their seed."""
    out = {}
    # sections: LINES <tag> A-B and GAPFILL <tag>
    for key in (f"--- LINES {tag} ", f"--- GAPFILL {tag} ---"):
        i = log.rfind(key)
        if i < 0:
            continue
        seg = log[i:]
        end = len(seg)
        for stop in ("--- LINES", "--- GAPFILL", "--- CHAIN", "--- COUNTS", "--- SUMMARIES", "TGT-END"):
            j = seg.find(stop, 1)
            if 0 <= j < end:
                end = j
        for ln in seg[:end].splitlines():
            j = ln.find("{")
            if j < 0:
                continue
            try:
                rec = json.loads(ln[j:])
            except Exception:
                continue
            if "seed" in rec:
                out[rec["seed"]] = ln[j:]
    return out

def load_store():
    try:
        with open(STORE) as f:
            raw = json.load(f)
        return {t: {int(k): v for k, v in d.items()} for t, d in raw.items()}
    except Exception:
        return {t: {} for t in TAGS}

def save_store(store):
    with open(STORE, "w") as f:
        json.dump(store, f)

def load_linepos():
    try:
        with open(LINEPOS) as f:
            return {k: int(v) for k, v in json.load(f).items()}
    except Exception:
        return {}

def save_linepos(lp):
    with open(LINEPOS, "w") as f:
        json.dump(lp, f)

def put_branch(path, data_bytes, msg):
    sha = None
    try:
        r = gh_api.get_file(OWNER, REPO, path, ref=BRANCH)
        if isinstance(r, dict):
            sha = r.get("sha")
    except Exception:
        sha = None
    gh_api.put_file(OWNER, REPO, path, data_bytes, msg, sha=sha, branch=BRANCH)
    b = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/sync%2Fh29")
    return b["commit"]["sha"][:8], b["commit"]["commit"]["author"]["date"]

def seed0_of(store, tag, summaries):
    if tag in summaries:
        return summaries[tag].get("seed0")
    seeds = sorted(store.get(tag, {}))
    return seeds[0] if seeds else None

def main():
    log_line("SNAPLOOP_START h29 Fable r1 (v3.3: raw 50-line windows + per-tag summaries + seed gapfill + two-phase Range claim + API economy + v3.4 small-window + v3.5 partial-log + v3.6 linepos-window + claim90s)")
    store = load_store()
    linepos = load_linepos()
    summaries = {}
    trip = 0
    gone_streak = 0
    while time.time() < DEADLINE:
        trip += 1
        t0 = time.time()
        marker = "S29SNAPFABLE" + time.strftime("%H%M%S")
        # focus: first incomplete tag (rotate by trip); one tag per trip
        incomplete = [t for t in TAGS
                      if not (seed0_of(store, t, summaries) is not None
                              and len(store.get(t, {})) == 200
                              and sorted(store[t]) == list(range(seed0_of(store, t, summaries),
                                                                seed0_of(store, t, summaries) + 200)))]
        focus = incomplete[(trip - 1) % len(incomplete)] if incomplete else TAGS[0]
        windows, gaps = {}, {}
        s0 = seed0_of(store, focus, summaries)
        have = sorted(store.get(focus, {}))
        if s0 is not None and have:
            gaps[focus] = [s for s in range(s0, have[-1] + 1) if s not in store[focus]][:2]
        else:
            gaps[focus] = []
        # v3.6: window walks FILE LINE NUMBERS via persisted linepos.
        # Old nxt = max_seed - s0 + 2 froze once stored seeds formed a
        # complete prefix with no gaps below max (window re-fetched
        # already-stored lines, gap list empty) - four tags stalled
        # 2026-11-13 ~02:58-03:42. Line numbers advance on every reached
        # trip regardless of new rows; gap fill covers the rest.
        lp = linepos.get(focus, 1)
        if lp > 200:
            windows[focus] = (1, 0)  # window exhausted; gap fill only
        else:
            windows[focus] = (lp, min(lp + WINDOW - 1, 200))
        try:
            dispatch(pull_script(marker, windows, gaps, focus))
        except Exception as e:
            log_line(f"trip={trip} dispatch error: {e}")
            time.sleep(30)
            continue
        log = claim(marker, not_before=t0)
        if log is None:
            log_line(f"trip={trip} NOT-CLAIMED marker={marker}")
            time.sleep(20)
            continue
        tgt_reached = "TGT-BEGIN" in log
        partial = tgt_reached and "TGT-END" not in log
        if partial:
            log_line(f"trip={trip} PARTIAL log (no TGT-END, banking what parsed)")
        target_ok = tgt_reached
        if not target_ok:
            gone_streak += 1
            log_line(f"trip={trip} TARGET-UNREACHABLE streak={gone_streak} counts={branch_counts}")
            if gone_streak >= 2:
                log_line(f"trip={trip} MACHINE_GONE final_counts={branch_counts}")
                return
        else:
            gone_streak = 0
            status_lines = [ln.split("Z ", 1)[-1] for ln in log.splitlines() if "CHAIN_" in ln]
            done_all = any("CHAIN_ALL_DONE" in s for s in status_lines)
            # summaries first (they carry seed0) -> per-tag files
            tgt_i = log.rfind("TGT-BEGIN")
            real_log = log[tgt_i:] if tgt_i >= 0 else log
            for ln in real_log.splitlines():
                if '"tag": "fabler1' in ln and '"mean"' in ln:
                    i = ln.find("{")
                    if i >= 0:
                        try:
                            rec = json.loads(ln[i:])
                            if rec.get("tag") in TAGS:
                                summaries[rec["tag"]] = rec
                                sha8, _ = put_branch(
                                    f"synced/h29/summaries/{rec['tag']}.json",
                                    (json.dumps(rec, ensure_ascii=False) + "\n").encode(),
                                    f"h29 summary {rec['tag']}")
                                log_line(f"trip={trip} summary {rec['tag']} -> summaries/ @{sha8}")
                        except Exception:
                            pass
            tag = focus
            new = extract_lines(log, tag)
            # seed-domain filter (war-room 01:12): drop any line whose seed
            # falls outside this tag's own [s0, s0+199] before it can land
            if s0 is not None:
                _bad = [s for s in new if not (s0 <= s <= s0 + 199)]
                for _s in _bad:
                    log_line(f"trip={trip} DROP out-of-domain seed {_s} from {tag}")
                    del new[_s]
            before = len(store.get(tag, {}))
            store.setdefault(tag, {}).update(new)
            save_store(store)
            # v3.6: advance the line window on every reached trip
            if windows[tag][1] >= windows[tag][0]:
                linepos[tag] = windows[tag][1] + 1
                save_linepos(linepos)
            s0 = seed0_of(store, tag, summaries)
            have = sorted(store.get(tag, {}))
            complete = s0 is not None and len(have) == 200 and have == list(range(s0, s0 + 200))
            if have:
                data = ("\n".join(store[tag][s] for s in have) + "\n").encode()
                open(os.path.join(SNAP, tag + ".jsonl"), "wb").write(data)
                if branch_counts.get(tag) != len(have):
                    try:
                        sha8, cdate = put_branch(f"synced/h29/bigeval_live/{tag}.jsonl", data,
                                                 f"h29 snap {tag} {len(have)} lines")
                        branch_counts[tag] = len(have)
                        log_line(f"trip={trip} {tag} {before}->{len(have)} complete={complete} -> sync/h29 @{sha8} status={status_lines[-1] if status_lines else '?'}")
                    except Exception as e:
                        log_line(f"trip={trip} {tag} {len(have)} put failed (retry next trip): {e}")
                else:
                    log_line(f"trip={trip} {tag} {len(have)} unchanged complete={complete}")
            else:
                log_line(f"trip={trip} {tag} 0 lines (window {windows[tag]})")
            if all(len(store.get(t, {})) == 200 for t in TAGS) and done_all:
                log_line(f"trip={trip} ALL_DONE counts={branch_counts}")
                return
        dt = time.time() - t0
        if dt < INTERVAL:
            time.sleep(INTERVAL - dt)
    log_line(f"DEADLINE counts={branch_counts}")

if __name__ == "__main__":
    main()
