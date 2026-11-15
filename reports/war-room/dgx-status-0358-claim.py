"""DGX status probe 0358 CLAIM-ONLY driver: re-claim the run dispatched at
04:00:32 CST after the first driver was SIGTERM-killed by a service restart.
Does NOT re-dispatch. Marker-only claim, per 2026-11-12 cross-talk lesson.
Polls at most 8 minutes; saves the claimed job log to dgx-status-0358.log."""
import time, os, sys
import urllib.request, urllib.error, http.client

sys.path.insert(0, os.path.expanduser("~/workspace"))
import gh_api

MARK = "DGXSTATUS0358A"
HF = os.path.expanduser(
    "~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files")
LOG = os.path.join(HF, "dgx-status-0358.log")
# Dispatch was 04:00:32 CST = 20:00:32Z on 2026-11-12. Only runs created at/after 20:00:00Z count.
CUTOFF = "2026-11-12T20:00:00Z"


def fetch_job_log(job_id):
    """Fetch a job log, tolerating truncated blob downloads: keep the longest
    partial body, then resume with an open-ended Range fetch from that point."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    url = f"{gh_api.API}/repos/robertyang87/carbon-league/actions/jobs/{job_id}/logs"
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "muse-agent"})
    gh_api.add_surrogate_to_request(req, "custom.github", allowed_hosts=["api.github.com"])
    try:
        with opener.open(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location")
        if not loc:
            raise
    best = ""
    for i in range(3):
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


t0 = time.time()
print("claim-only start", time.strftime("%H:%M:%S"), "cutoff", CUTOFF, flush=True)
deadline = time.time() + 480
tried = {}
result = None
while time.time() < deadline:
    time.sleep(30)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
    except Exception as e:
        print("list error:", e, flush=True)
        continue
    for run in r.get("workflow_runs", []):
        rid = run["id"]
        if run["status"] != "completed":
            continue
        if run["created_at"] < CUTOFF:
            continue
        if tried.get(rid, 0) >= 4:
            continue
        tried[rid] = tried.get(rid, 0) + 1
        print("trying run", rid, "round", tried[rid], "conclusion", run.get("conclusion"), flush=True)
        try:
            jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
        except Exception as e:
            print("jobs error:", e, flush=True)
            continue
        for j in jobs.get("jobs", []):
            try:
                log = fetch_job_log(j["id"])
            except Exception as e:
                print("log error:", e, flush=True)
                continue
            print(f"  job {j['id']} log bytes={len(log)}", flush=True)
            if isinstance(log, str) and MARK in log:
                print("CLAIMED run", rid, "job", j["id"], "conclusion", run.get("conclusion"), flush=True)
                i0 = log.find(MARK)
                print(log[max(0, i0 - 200):i0 + 12000], flush=True)
                with open(LOG, "w") as f:
                    f.write(log)
                result = log
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time() - t0), "claimed" if result else "NOT-CLAIMED", flush=True)
