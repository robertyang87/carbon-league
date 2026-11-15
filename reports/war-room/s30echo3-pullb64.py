"""Archival pull of both echo3 live files from spot-30 via gzip+base64
ranged pulls (each b64 line is index-tagged, so dropped/corrupt lines
are located exactly and re-pulled). Assembles, gunzips, md5-verifies
against the on-machine md5, saves to hidden_files/s30echo3-readback-*,
and recomputes per-tag stats locally."""
import base64, gzip, hashlib, json, os, re, statistics as st, sys, time
sys.path.insert(0, "/home/hatch/workspace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gh_api
import urllib.error, urllib.request
from datetime import datetime, timezone

def fetch_log_quick(job_id, t1=15, t2=25):
    """Bounded log fetch: short timeouts so one slow blob cannot eat the
    whole claim window."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    url = f"{gh_api.API}/repos/{OWNER}/{REPO}/actions/jobs/{job_id}/logs"
    r = urllib.request.Request(url, method="GET", headers={"User-Agent": "muse-agent"})
    gh_api.add_surrogate_to_request(r, "custom.github", allowed_hosts=["api.github.com"])
    try:
        with opener.open(r, timeout=t1) as resp:
            return resp.read().decode("utf-8", errors="replace"), "full"
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location")
        if not loc:
            raise
    r2 = urllib.request.Request(loc, method="GET", headers={"User-Agent": "muse-agent"})
    try:
        with urllib.request.urlopen(r2, timeout=t2) as resp2:
            return resp2.read().decode("utf-8", errors="replace"), "full"
    except Exception as e:
        partial = getattr(e, "partial", b"") or b""
        return partial.decode("utf-8", errors="replace"), f"partial:{type(e).__name__}"

HF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"
INST = "i-uf6ak9krs1kt71y2unce"
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+")
LINE_RE = re.compile(r"^(\d+)\|([A-Za-z0-9+/=]{1,76})$")
TAGS = {
    "echo3_m2": ("7e49cb1945011219491d49d0a8e100cc", 996000),
    "echo3_echo2": ("0ba321f989eee75006e7e594dcca63d2", 999000),
}
CHUNK = 300

def log_line(s):
    print(f"{time.strftime('%H:%M:%S')} {s}", flush=True)

def dispatch(script):
    gh_api.dispatch(OWNER, REPO, WF, ref="main",
                    inputs={"instance_id": INST, "region": "cn-shanghai",
                            "command_b64": base64.b64encode(script.encode()).decode()})
    return datetime.now(timezone.utc)

def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def claim(marker, t_disp, timeout_s=260):
    deadline = time.time() + timeout_s
    seen = set()
    while time.time() < deadline:
        time.sleep(10)
        try:
            r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30")
        except Exception as e:
            log_line(f"list error {e}")
            continue
        for run in r.get("workflow_runs", []):
            rid = run["id"]
            if rid in seen:
                continue
            try:
                created = parse_ts(run["created_at"])
            except Exception:
                continue
            if created <= t_disp or run["status"] != "completed":
                continue
            seen.add(rid)
            try:
                jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs")
            except Exception:
                continue
            for j in jobs.get("jobs", []):
                try:
                    log, how = fetch_log_quick(j["id"])
                except Exception:
                    continue
                if marker in log:
                    return log
    return None

def pull_range(tag, a, b):
    marker = "S30E3PB" + time.strftime("%H%M%S") + str(a % 997)
    script = f"""MARK={marker}
echo "$MARK PAYLOAD-BEGIN"
gzip -n -c /root/carbon/data/bigeval_live/{tag}.jsonl | base64 -w 76 | sed -n '{a},{b}p' | awk -v o={a} '{{printf "%d|%s\\n", o+NR-1, $0}}'
echo "$MARK PAYLOAD-END"
"""
    t = dispatch(script)
    log = claim(marker, t)
    got = {}
    if log is None:
        log_line(f"{tag} range {a}-{b} NOT-CLAIMED")
        return got
    text = "\n".join(TS_RE.sub("", ln) for ln in log.splitlines())
    i0, i1 = text.find(marker + " PAYLOAD-BEGIN"), text.find(marker + " PAYLOAD-END")
    if i0 < 0 or i1 < 0:
        log_line(f"{tag} range {a}-{b} payload markers missing")
        return got
    for ln in text[i0:i1].splitlines():
        m = LINE_RE.match(ln.strip())
        if m:
            got[int(m.group(1))] = m.group(2)
    log_line(f"{tag} range {a}-{b} got {len(got)}/{b - a + 1}")
    return got

def pull_count(tag):
    marker = "S30E3PC" + time.strftime("%H%M%S")
    script = f"""MARK={marker}
echo "$MARK PAYLOAD-BEGIN"
gzip -n -c /root/carbon/data/bigeval_live/{tag}.jsonl | base64 -w 76 | wc -l
md5sum /root/carbon/data/bigeval_live/{tag}.jsonl
echo "$MARK PAYLOAD-END"
"""
    t = dispatch(script)
    log = claim(marker, t)
    if log is None:
        return None, None
    text = "\n".join(TS_RE.sub("", ln) for ln in log.splitlines())
    i0, i1 = text.find(marker + " PAYLOAD-BEGIN"), text.find(marker + " PAYLOAD-END")
    payload = text[i0:i1]
    cnt = re.search(r"\n\s*(\d+)\s*\n", payload)
    md5 = re.search(r"([0-9a-f]{32})\s+/root", payload)
    return (int(cnt.group(1)) if cnt else None), (md5.group(1) if md5 else None)

for tag, (exp_md5, seed0) in TAGS.items():
    total, md5_now = pull_count(tag)
    log_line(f"{tag}: b64 lines={total} md5={md5_now} expected_md5={exp_md5}")
    assert total and md5_now == exp_md5, f"{tag} count/md5 problem"
    state_f = f"{HF}/s30echo3-pullb64-state-{tag}.json"
    lines = {}
    if os.path.exists(state_f):
        lines = {int(k): v for k, v in json.load(open(state_f)).items()}
        log_line(f"{tag}: restored {len(lines)} lines from state")
    def save_state():
        json.dump({str(k): v for k, v in lines.items()}, open(state_f, "w"))
    a = 1
    while a <= total:
        b = min(total, a + CHUNK - 1)
        if all(i in lines for i in range(a, b + 1)):
            a = b + 1
            continue
        lines.update(pull_range(tag, a, b))
        save_state()
        a = b + 1
    # gap rounds
    for rnd in range(6):
        missing = [i for i in range(1, total + 1) if i not in lines]
        if not missing:
            break
        log_line(f"{tag}: gap round {rnd} missing {len(missing)}")
        # group missing into contiguous-ish ranges of <=120
        groups = []
        cur = [missing[0]]
        for x in missing[1:]:
            if x - cur[-1] <= 3 and len(cur) < 120:
                cur.append(x)
            else:
                groups.append(cur); cur = [x]
        groups.append(cur)
        for g in groups:
            got = pull_range(tag, g[0], g[-1])
            for k, v in got.items():
                if k in missing:
                    lines[k] = v
            save_state()
    missing = [i for i in range(1, total + 1) if i not in lines]
    log_line(f"{tag}: final missing {len(missing)} of {total}")
    if missing:
        log_line(f"{tag}: INCOMPLETE, first missing {missing[:20]}")
        continue
    blob = base64.b64decode("".join(lines[i] for i in range(1, total + 1)))
    raw = gzip.decompress(blob)
    md5 = hashlib.md5(raw).hexdigest()
    out = f"{HF}/s30echo3-readback-{tag}.jsonl"
    open(out, "wb").write(raw)
    rows = [json.loads(ln) for ln in raw.decode().splitlines() if ln.strip()]
    m = [r["margin"] for r in rows if r.get("margin") is not None]
    err = sum(1 for r in rows if r.get("err"))
    seeds = sorted(r["seed"] for r in rows)
    se = st.stdev(m) / len(m) ** .5 if len(m) > 1 else 0
    log_line(f"{tag}: SAVED {out} bytes={len(raw)} md5={md5} match={md5 == exp_md5} "
             f"rows={len(rows)} wins={sum(x > 0 for x in m)} mean={round(st.mean(m), 1)} "
             f"se={round(se, 1)} lo2se={round(st.mean(m) - 2 * se, 1)} err={err} "
             f"seeds={len(set(seeds))} range={seeds[0]}-{seeds[-1]}")
log_line("PULL-ALL-DONE")
