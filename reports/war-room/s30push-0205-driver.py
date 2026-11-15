"""spot-30 exam-stack push driver: ONE launch dispatch + claim, then poll loop.
Clone of hakimi2-launch-driver2.py: claim loop only looks at runs created
AFTER each dispatch; log fetch tolerant (redirect/partial handling).
Fix vs driver2: a run seen while still running is NOT burned -- it stays
eligible until it completes (driver2 marked it seen and could skip it).
"""
import sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api, base64, time, urllib.error, urllib.request
from datetime import datetime, timezone

HF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
MARK = "S30PUSH0205"
PMARK = "S30POLL0205"
L20 = "i-uf62k0o6zbe6j7zxnzg5"  # gpu-l20-5
OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"

def fetch_log_tolerant(owner, repo, job_id):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    url = f"{gh_api.API}/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"
    r = urllib.request.Request(url, method="GET", headers={"User-Agent": "muse-agent"})
    gh_api.add_surrogate_to_request(r, "custom.github", allowed_hosts=["api.github.com"])
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

def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def dispatch(script_path):
    script = open(script_path).read()
    print("dispatch", script_path, "bytes:", len(script), flush=True)
    resp = gh_api.dispatch(OWNER, REPO, WF, ref="main",
        inputs={"instance_id": L20, "region": "cn-shanghai",
                "command_b64": base64.b64encode(script.encode()).decode()})
    print("dispatch resp:", resp, flush=True)
    return resp

def claim(marker, t_dispatch, timeout_s):
    """Return (log_slice, run_id, conclusion) or (None, None, None)."""
    deadline = time.time() + timeout_s
    seen_done = set()
    seen_running = set()
    while time.time() < deadline:
        time.sleep(25)
        try:
            r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30")
        except Exception as e:
            print("list error:", e, flush=True); continue
        for run in r.get("workflow_runs", []):
            rid = run["id"]
            if rid in seen_done:
                continue
            try:
                created = parse_ts(run["created_at"])
            except Exception:
                created = None
            if created is not None and created <= t_dispatch:
                continue
            if run["status"] != "completed":
                if rid not in seen_running:
                    seen_running.add(rid)
                    print("new run", rid, "status", run["status"], "created", run["created_at"], flush=True)
                continue
            seen_done.add(rid)
            print("completed run", rid, "conclusion", run.get("conclusion"), "created", run["created_at"], flush=True)
            try:
                jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs")
            except Exception as e:
                print("jobs error:", e, flush=True); continue
            for j in jobs.get("jobs", []):
                best = ""
                for attempt in range(5):
                    try:
                        log, n, how = fetch_log_tolerant(OWNER, REPO, j["id"])
                    except Exception as e:
                        print("log error:", e, flush=True); continue
                    if len(log) > len(best):
                        best = log
                    print(rid, j["id"], "attempt", attempt, "bytes", n, how, "marker" if marker in log else "no-marker", flush=True)
                    if marker in log:
                        break
                if best and marker in best:
                    print("CLAIMED run", rid, "job", j["id"], "conclusion", run.get("conclusion"), flush=True)
                    i0 = best.find(marker)
                    return best[max(0, i0 - 200):i0 + 60000], rid, run.get("conclusion")
    return None, None, None

def main():
    t0 = time.time()
    try:
        rl = gh_api.req("GET", "/rate_limit")
        print("rate_limit core:", rl["resources"]["core"], flush=True)
    except Exception as e:
        print("rate_limit check failed:", e, flush=True)

    dispatch(f"{HF}/s30push-0205.sh")
    t_disp = datetime.now(timezone.utc)
    print("launched at", time.strftime("%H:%M:%S"), flush=True)
    log, rid, concl = claim(MARK, t_disp, 600)
    if log is None:
        print("RESULT: LAUNCH-NOT-CLAIMED elapsed", round(time.time() - t0), flush=True)
        return
    open(f"{HF}/s30push-0205.log", "w").write(f"run={rid} conclusion={concl}\n" + log)
    print(log, flush=True)
    if "SSH-NOT-READY" in log:
        print("RESULT: SSH-NOT-READY", flush=True)
        return
    if "LAUNCHED" not in log:
        print("RESULT: LAUNCH-UNCLEAR", flush=True)
        return
    final = None
    while time.time() - t0 < 1800:
        time.sleep(75)
        dispatch(f"{HF}/s30poll-0205.sh")
        t_pdisp = datetime.now(timezone.utc)
        plog, prid, pconcl = claim(PMARK, t_pdisp, 300)
        if plog is None:
            print("POLL-NOT-CLAIMED", flush=True)
            continue
        open(f"{HF}/s30poll-0205.log", "w").write(f"run={prid} conclusion={pconcl}\n" + plog)
        print("--- POLL", time.strftime("%H:%M:%S"), "---", flush=True)
        print(plog[-4000:], flush=True)
        if "WORKER-DONE" in plog:
            final = plog
            break
    if final:
        print("RESULT: DONE", "MD5-ALL-MATCH" if "MD5-ALL-MATCH" in final else "MD5-NOT-ALL-MATCH",
              "elapsed", round(time.time() - t0), flush=True)
    else:
        print("RESULT: TIMEOUT elapsed", round(time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
