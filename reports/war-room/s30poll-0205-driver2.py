"""spot-30 push lean poll: ONE poll dispatch + lean claim.
Fix vs s30push-0205-driver.py: the claim loop fetched every non-matching
run's full log up to 5 times (each ~42KB), burning the 300s budget in the
crowded post-reset claim channel. Here: one fetch per job; a "full" log
without the marker is final -- retry only partial/truncated fetches.
"""
import sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api, base64, time, urllib.error, urllib.request
from datetime import datetime, timezone

HF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
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

def main():
    t0 = time.time()
    try:
        rl = gh_api.req("GET", "/rate_limit")
        print("rate_limit core:", rl["resources"]["core"], flush=True)
    except Exception as e:
        print("rate_limit check failed:", e, flush=True)

    script = open(f"{HF}/s30poll-0205.sh").read()
    gh_api.dispatch(OWNER, REPO, WF, ref="main",
        inputs={"instance_id": L20, "region": "cn-shanghai",
                "command_b64": base64.b64encode(script.encode()).decode()})
    t_disp = datetime.now(timezone.utc)
    print("polled at", time.strftime("%H:%M:%S"), flush=True)

    deadline = time.time() + 600
    seen_done = set()
    seen_running = set()
    result = None
    while time.time() < deadline:
        time.sleep(20)
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
            if created is not None and created <= t_disp:
                continue
            if run["status"] != "completed":
                if rid not in seen_running:
                    seen_running.add(rid)
                    print("new run", rid, "status", run["status"], flush=True)
                continue
            seen_done.add(rid)
            try:
                jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs")
            except Exception as e:
                print("jobs error:", e, flush=True); continue
            for j in jobs.get("jobs", []):
                best = ""
                for attempt in range(3):
                    try:
                        log, n, how = fetch_log_tolerant(OWNER, REPO, j["id"])
                    except Exception as e:
                        print("log error:", e, flush=True); break
                    if len(log) > len(best):
                        best = log
                    print(rid, j["id"], "attempt", attempt, "bytes", n, how,
                          "marker" if PMARK in log else "no-marker", flush=True)
                    if PMARK in log or how == "full":
                        break
                if best and PMARK in best:
                    print("CLAIMED run", rid, "job", j["id"], "conclusion", run.get("conclusion"), flush=True)
                    i0 = best.find(PMARK)
                    result = best[max(0, i0 - 200):i0 + 60000]
                    break
            if result:
                break
        if result:
            break
    if result is None:
        print("RESULT: POLL-NOT-CLAIMED elapsed", round(time.time() - t0), flush=True)
        return
    open(f"{HF}/s30poll-0205.log", "w").write(result)
    print(result, flush=True)
    tail = result[-4000:]
    print("--- TAIL ---", flush=True)
    print(tail, flush=True)
    if "WORKER-DONE" in result:
        print("RESULT: DONE", "MD5-ALL-MATCH" if "MD5-ALL-MATCH" in result else "MD5-NOT-ALL-MATCH",
              "elapsed", round(time.time() - t0), flush=True)
    else:
        print("RESULT: WORKER-STILL-RUNNING elapsed", round(time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
