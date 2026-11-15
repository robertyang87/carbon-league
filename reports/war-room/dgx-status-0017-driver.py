import gh_api, base64, time, os, urllib.request, urllib.error, http.client
MARK = "DGXSTATUS0017A"
LOG = os.path.expanduser("~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-status-0017.log")
os.chdir(os.path.expanduser("~/workspace"))
script = open("goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-status-0017.sh").read()

def fetch_job_log_tolerant(job_id, attempts=3):
    """Like gh_api.get_job_logs but tolerates truncated downloads: keeps the
    longest partial body across attempts instead of raising."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    url = f"{gh_api.API}/repos/robertyang87/carbon-league/actions/jobs/{job_id}/logs"
    r = urllib.request.Request(url, method="GET", headers={"User-Agent": "muse-agent"})
    gh_api.add_surrogate_to_request(r, "custom.github", allowed_hosts=["api.github.com"])
    loc = None
    try:
        with opener.open(r, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location")
        if not loc:
            raise
    best = ""
    for i in range(attempts):
        try:
            r2 = urllib.request.Request(loc, method="GET", headers={"User-Agent": "muse-agent"})
            with urllib.request.urlopen(r2, timeout=60) as resp2:
                data = resp2.read().decode("utf-8", errors="replace")
            if len(data) > len(best):
                best = data
            if MARK in best:
                break
        except http.client.IncompleteRead as e:
            part = e.partial.decode("utf-8", errors="replace") if isinstance(e.partial, bytes) else str(e.partial)
            print(f"  truncated attempt {i+1}: got {len(part)} bytes", flush=True)
            if len(part) > len(best):
                best = part
        except Exception as e:
            print(f"  fetch attempt {i+1} error: {e}", flush=True)
    if MARK not in best and best:
        # 2026-11-12 23:51 lesson: blob download dies at a fixed ~15.3KB point.
        # Open-ended Range fetch from the truncation point returns the tail.
        try:
            r3 = urllib.request.Request(loc, method="GET",
                headers={"User-Agent": "muse-agent", "Range": f"bytes={len(best)}-"})
            with urllib.request.urlopen(r3, timeout=60) as resp3:
                tail = resp3.read().decode("utf-8", errors="replace")
            print(f"  range tail: +{len(tail)} bytes from {len(best)}", flush=True)
            best = best + tail
        except Exception as e:
            print(f"  range fetch error: {e}", flush=True)
    return best

t0 = time.time()
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": "i-uf62k0o6zbe6j7zxnzg5", "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched at", time.strftime("%H:%M:%S"), flush=True)
deadline = time.time() + 600
seen_claimed = set()
tried = {}
result = None
while time.time() < deadline:
    time.sleep(30)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
    except Exception as e:
        print("list error:", e, flush=True); continue
    for run in r.get("workflow_runs", []):
        rid = run["id"]
        if rid in seen_claimed or run["status"] != "completed":
            continue
        if run["created_at"] < time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 - 120)):
            continue
        if tried.get(rid, 0) >= 4:
            continue
        tried[rid] = tried.get(rid, 0) + 1
        print("trying run", rid, "round", tried[rid], "conclusion", run.get("conclusion"), flush=True)
        try:
            jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
        except Exception as e:
            print("jobs error:", e, flush=True); continue
        for j in jobs.get("jobs", []):
            try:
                log = fetch_job_log_tolerant(j["id"])
            except Exception as e:
                print("log error:", e, flush=True); continue
            print(f"  job {j['id']} log bytes={len(log)}", flush=True)
            if isinstance(log, str) and MARK in log:
                print("CLAIMED run", rid, "job", j["id"], "conclusion", run.get("conclusion"), flush=True)
                i0 = log.find(MARK)
                print(log[max(0, i0-200):i0+12000], flush=True)
                with open(LOG, "w") as f:
                    f.write(log)
                result = log
                seen_claimed.add(rid)
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time()-t0), "claimed" if result else "NOT-CLAIMED", flush=True)
