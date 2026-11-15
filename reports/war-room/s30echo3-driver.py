"""Generic cloud-assistant driver for the s30echo3 line.

Usage: s30echo3-driver.py <script> <instance_id> <marker> <timeout_s> <out_log>
Dispatches the script via cloud-assistant.yml (command_b64), then claims
ONLY the run created after this dispatch whose job log contains the
marker. Log fetch is tolerant: keeps the longest copy across attempts.
Prints the payload between '<marker> PAYLOAD-BEGIN' and
'<marker> PAYLOAD-END' (Actions log lines carry a timestamp prefix, so
extraction is by substring).
"""
import sys, os, base64, time, urllib.error, urllib.request
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
from datetime import datetime, timezone

OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"

def fetch_log_tolerant(job_id):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    url = f"{gh_api.API}/repos/{OWNER}/{REPO}/actions/jobs/{job_id}/logs"
    r = urllib.request.Request(url, method="GET", headers={"User-Agent": "muse-agent"})
    gh_api.add_surrogate_to_request(r, "custom.github", allowed_hosts=["api.github.com"])
    try:
        with opener.open(r, timeout=30) as resp:
            data = resp.read()
            return data.decode("utf-8", errors="replace"), "full"
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location")
        if not loc:
            raise
    r2 = urllib.request.Request(loc, method="GET", headers={"User-Agent": "muse-agent"})
    try:
        with urllib.request.urlopen(r2, timeout=60) as resp2:
            data = resp2.read()
            return data.decode("utf-8", errors="replace"), "full"
    except Exception as e:
        partial = getattr(e, "partial", b"") or b""
        return partial.decode("utf-8", errors="replace"), f"partial:{type(e).__name__}"

def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def main():
    sh_file, inst, MARK, tmo, out_log = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    script = open(sh_file).read()
    assert MARK in script, "marker not in script"
    try:
        rl = gh_api.req("GET", "/rate_limit")
        print("rate_limit core remaining:", rl["resources"]["core"]["remaining"], flush=True)
    except Exception as e:
        print("rate_limit check failed:", e, flush=True)
    gh_api.dispatch(OWNER, REPO, WF, ref="main",
                    inputs={"instance_id": inst, "region": "cn-shanghai",
                            "command_b64": base64.b64encode(script.encode()).decode()})
    t_disp = datetime.now(timezone.utc)
    print("dispatched", inst, MARK, "at", time.strftime("%H:%M:%S"), flush=True)
    deadline = time.time() + tmo
    seen_done, seen_running = set(), set()
    result = None
    while time.time() < deadline:
        time.sleep(20)
        try:
            r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30")
        except Exception as e:
            print("list error:", e, flush=True)
            if "403" in str(e) or "rate" in str(e).lower():
                print("RATE-LIMITED: stopping silently per discipline", flush=True)
                break
            continue
        for run in r.get("workflow_runs", []):
            rid = run["id"]
            if rid in seen_done:
                continue
            try:
                created = parse_ts(run["created_at"])
            except Exception:
                continue
            if created <= t_disp:
                continue
            if run["status"] != "completed":
                if rid not in seen_running:
                    seen_running.add(rid)
                    print("new run", rid, run["status"], run["created_at"], flush=True)
                continue
            seen_done.add(rid)
            print("completed run", rid, run.get("conclusion"), run["created_at"], flush=True)
            try:
                jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs")
            except Exception as e:
                print("jobs error:", e, flush=True); continue
            for j in jobs.get("jobs", []):
                best = ""
                for attempt in range(4):
                    try:
                        log, how = fetch_log_tolerant(j["id"])
                    except Exception as e:
                        print("log error:", e, flush=True); continue
                    if len(log) > len(best):
                        best = log
                    print(rid, "attempt", attempt, "bytes", len(log), how,
                          "marker" if MARK in log else "no-marker", flush=True)
                    if MARK in log:
                        break
                if best and MARK in best:
                    print("CLAIMED run", rid, "conclusion", run.get("conclusion"), flush=True)
                    open(out_log, "w").write(f"run={rid} conclusion={run.get('conclusion')}\n" + best)
                    i0 = best.find(MARK + " PAYLOAD-BEGIN")
                    i1 = best.find(MARK + " PAYLOAD-END")
                    if i0 >= 0 and i1 > i0:
                        print(best[i0:i1 + len(MARK) + 12], flush=True)
                    else:
                        print("PAYLOAD MARKERS NOT FOUND; log tail:", flush=True)
                        print(best[-6000:], flush=True)
                    result = best
                    break
            if result:
                break
        if result:
            break
    print("RESULT:", "CLAIMED" if result else "NOT-CLAIMED", flush=True)

if __name__ == "__main__":
    main()
