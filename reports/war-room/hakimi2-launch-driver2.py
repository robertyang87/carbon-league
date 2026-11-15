"""Hakimi v2 pro Studio launch driver, TRIP 2: ONE dispatch + claim.
Fix vs trip-1 driver: claim loop only looks at runs created AFTER this
dispatch (trip 1 churned through ~40 other lines' runs and ~130 log fetches
before finding its own run at 729s elapsed).
"""
import sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api, base64, time, urllib.error, urllib.request
from datetime import datetime, timezone

MARK = "HAKIMI2LCH2"
HF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
script = open(f"{HF}/hakimi2-studio-launch2.sh").read()
print("script bytes:", len(script), "b64 bytes:", len(base64.b64encode(script.encode())), flush=True)

try:
    rl = gh_api.req("GET", "/rate_limit")
    print("rate_limit core:", rl["resources"]["core"], flush=True)
except Exception as e:
    print("rate_limit check failed:", e, flush=True)

def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

t_dispatch = datetime.now(timezone.utc)
t0 = time.time()
t_dispatch_s = time.strftime("%H:%M:%S")
resp = gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": "i-uf62k0o6zbe6j7zxnzg5", "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched at", t_dispatch_s, "resp:", resp, flush=True)

def fetch_log_tolerant(owner, repo, job_id):
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

deadline = time.time() + 600
seen_runs = set()
result = None
run_id = job_id = conclusion = None
while time.time() < deadline:
    time.sleep(25)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
    except Exception as e:
        print("list error:", e, flush=True); continue
    for run in r.get("workflow_runs", []):
        rid = run["id"]
        if rid in seen_runs:
            continue
        seen_runs.add(rid)
        try:
            created = parse_ts(run["created_at"])
        except Exception:
            created = None
        if created is not None and created <= t_dispatch:
            continue
        print("new run", rid, "status", run["status"], "created", run["created_at"], flush=True)
        if run["status"] != "completed":
            continue
        try:
            jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
        except Exception as e:
            print("jobs error:", e, flush=True); continue
        for j in jobs.get("jobs", []):
            best = ""
            for attempt in range(5):
                try:
                    log, n, how = fetch_log_tolerant("robertyang87", "carbon-league", j["id"])
                except Exception as e:
                    print("log error:", e, flush=True); continue
                if len(log) > len(best):
                    best = log
                print(rid, j["id"], "attempt", attempt, "bytes", n, how, "marker" if MARK in log else "no-marker", flush=True)
                if MARK in log:
                    break
            if best and MARK in best:
                print("CLAIMED run", rid, "job", j["id"], "conclusion", run.get("conclusion"), flush=True)
                i0 = best.find(MARK)
                print(best[max(0, i0-200):i0+16000], flush=True)
                result = best
                run_id, job_id, conclusion = rid, j["id"], run.get("conclusion")
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time()-t0), "claimed" if result else "NOT-CLAIMED", flush=True)
if result:
    open(f"{HF}/hakimi2-launch2-claim.log", "w").write(
        f"run={run_id} job={job_id} conclusion={conclusion} dispatched_at={t_dispatch_s}\n" + result)
    print("saved claim log", flush=True)
