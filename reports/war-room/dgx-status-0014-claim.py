import gh_api, time, os, urllib.request, urllib.error, http.client
MARK = "DGXSTATUS0014A"
LOG = os.path.expanduser("~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-status-0014.log")
def fetch_job_log_tolerant(job_id, attempts=3):
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

# dispatched 00:30:01 CST = 2026-11-12 16:30:01 UTC
r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
cands = [run for run in r.get("workflow_runs", []) if run["created_at"] >= "2026-11-12T16:29:00Z"]
print("candidate runs:", [(x["id"], x["created_at"], x["status"], x.get("conclusion")) for x in cands], flush=True)
found = None
for run in cands:
    if run["status"] != "completed":
        print("run", run["id"], "not completed:", run["status"], flush=True); continue
    jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{run['id']}/jobs")
    for j in jobs.get("jobs", []):
        log = fetch_job_log_tolerant(j["id"])
        print(f"  run {run['id']} job {j['id']} log bytes={len(log)}", flush=True)
        if isinstance(log, str) and MARK in log:
            i0 = log.find(MARK)
            print(log[max(0, i0-200):i0+12000], flush=True)
            with open(LOG, "w") as f:
                f.write(log)
            found = (run["id"], j["id"], run.get("conclusion"))
            break
    if found:
        break
print("CLAIMED" if found else "NOT-CLAIMED", found, flush=True)
