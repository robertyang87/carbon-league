import sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api, urllib.error, urllib.request, time
MARK = "STUDIOSTAT2357IT320"

def fetch_log_tolerant(owner, repo, job_id):
    """Same 302 handling as gh_api.get_job_logs, but keep partial bytes on IncompleteRead."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    url = f"{gh_api.API}/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"
    r = urllib.request.Request(url, method="GET", headers={"User-Agent": "muse-agent"})
    gh_api.add_surrogate_to_request(r, "custom.github", allowed_hosts=["api.github.com"])
    loc = None
    try:
        with opener.open(r, timeout=30) as resp:
            data = resp.read()
            return data.decode("utf-8", errors="replace"), len(data), "full"
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location")
        if not loc:
            raise
    r2 = urllib.request.Request(loc, method="GET", headers={"User-Agent": "muse-agent"})
    try:
        with urllib.request.urlopen(r2, timeout=60) as resp2:
            data = resp2.read()
            return data.decode("utf-8", errors="replace"), len(data), "full"
    except Exception as e:
        partial = getattr(e, "partial", b"") or b""
        return partial.decode("utf-8", errors="replace"), len(partial), f"partial:{type(e).__name__}"

r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
cands = [run for run in r.get("workflow_runs", [])
         if run["created_at"] >= "2026-11-12T15:58:30Z" and run["status"] == "completed"]
print("candidate completed runs since dispatch:", len(cands), flush=True)
found = None
for run in cands:
    jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{run['id']}/jobs")
    for j in jobs.get("jobs", []):
        try:
            log, n, how = fetch_log_tolerant("robertyang87", "carbon-league", j["id"])
        except Exception as e:
            print(run["id"], j["id"], "fetch error", e, flush=True)
            continue
        print(run["id"], j["id"], run.get("conclusion"), "bytes", n, how, "marker" if MARK in log else "no-marker", flush=True)
        if MARK in log:
            i0 = log.find(MARK)
            print(log[max(0, i0-200):i0+16000], flush=True)
            found = (run["id"], j["id"])
            break
    if found:
        break
print("FOUND" if found else "NOT-FOUND", found, flush=True)
