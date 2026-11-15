import gh_api, base64, time, os, sys, urllib.request, urllib.error
sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request

MARK = "PBSUM0013A"
HF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
os.chdir(os.path.expanduser("~/workspace"))
script = open(HF + "/pbsum-0013.sh").read()
t0 = time.time()
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": "i-uf62k0o6zbe6j7zxnzg5", "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched at", time.strftime("%H:%M:%S"), flush=True)

def blob_url(job_id):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    url = f"https://api.github.com/repos/robertyang87/carbon-league/actions/jobs/{job_id}/logs"
    r = urllib.request.Request(url, method="GET", headers={"User-Agent": "muse-agent"})
    add_surrogate_to_request(r, "custom.github", allowed_hosts=["api.github.com"])
    try:
        with opener.open(r, timeout=30) as resp:
            return None  # no redirect: treat as unavailable for range path
    except urllib.error.HTTPError as e:
        return e.headers.get("Location")

def fetch_ranged(job_id, seg=12000, max_bytes=400000):
    loc = blob_url(job_id)
    if not loc:
        return None
    out = b""
    start = 0
    while start < max_bytes:
        got = None
        for attempt in range(4):
            try:
                r2 = urllib.request.Request(loc, method="GET",
                    headers={"User-Agent": "muse-agent",
                             "Range": f"bytes={start}-{start + seg - 1}"})
                with urllib.request.urlopen(r2, timeout=60) as resp2:
                    got = resp2.read()
                break
            except Exception as e:
                print("range fetch error", start, repr(e)[:80], flush=True)
                time.sleep(3)
        if got is None:
            return out.decode("utf-8", errors="replace") if out else None
        out += got
        if len(got) < seg:
            break
        start += len(got)
    return out.decode("utf-8", errors="replace")

deadline = time.time() + 1500
tried_whole = set()
result = None
while time.time() < deadline and not result:
    time.sleep(20)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
    except Exception as e:
        print("list error:", e, flush=True); continue
    for run in r.get("workflow_runs", []):
        rid = run["id"]
        if run["status"] != "completed":
            continue
        if run["created_at"] < time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 - 120)):
            continue
        try:
            jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
        except Exception as e:
            print("jobs error:", e, flush=True); continue
        for j in jobs.get("jobs", []):
            jid = j["id"]
            log = None
            if jid not in tried_whole:
                tried_whole.add(jid)
                try:
                    log = gh_api.get_job_logs("robertyang87", "carbon-league", jid)
                except Exception as e:
                    print("whole-log error, falling to ranged:", repr(e)[:80], flush=True)
                    log = None
            if log is None or MARK not in log:
                try:
                    log2 = fetch_ranged(jid)
                except Exception as e:
                    print("ranged error:", repr(e)[:80], flush=True); log2 = None
                if log2 and MARK in log2:
                    log = log2
            if isinstance(log, str) and MARK in log:
                print("CLAIMED run", rid, "job", jid, "conclusion", run.get("conclusion"), flush=True)
                i0 = log.find(MARK)
                segtxt = log[max(0, i0 - 200):i0 + 30000]
                open(HF + "/pbsum-0013-recovered.txt", "w").write(segtxt)
                print(segtxt, flush=True)
                result = log
                break
        if result:
            break
print("elapsed", round(time.time() - t0), "claimed" if result else "NOT-CLAIMED", flush=True)
