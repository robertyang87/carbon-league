"""Claim-only pass for the s30echo3 line.

Usage: s30echo3-claim.py <marker> <t0_iso> <t1_iso> <out_log>
Scans cloud-assistant runs whose created_at falls in [t0, t1] (paged,
per_page=30), fetches each completed run's job log ONCE (full copy), and
claims the one containing the marker. No dispatch.
"""
import sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
from datetime import datetime, timezone
from s30echo3_driver_lib import fetch_log_tolerant

OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"

def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def main():
    MARK, t0s, t1s, out_log = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    t0, t1 = parse_ts(t0s), parse_ts(t1s)
    candidates = []
    for page in (1, 2, 3):
        r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30&page={page}")
        runs = r.get("workflow_runs", [])
        if not runs:
            break
        for run in runs:
            created = parse_ts(run["created_at"])
            if t0 <= created <= t1:
                candidates.append(run)
        if parse_ts(runs[-1]["created_at"]) < t0:
            break
    print("candidates in window:", [(c["id"], c["created_at"], c["status"]) for c in candidates], flush=True)
    for run in candidates:
        if run["status"] != "completed":
            print("skip not completed", run["id"], run["status"], flush=True)
            continue
        jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{run['id']}/jobs")
        for j in jobs.get("jobs", []):
            log, how = fetch_log_tolerant(j["id"])
            print(run["id"], "job", j["id"], "bytes", len(log), how,
                  "MARKER" if MARK in log else "no-marker", flush=True)
            if MARK in log:
                open(out_log, "w").write(f"run={run['id']} conclusion={run.get('conclusion')}\n" + log)
                i0 = log.find(MARK + " PAYLOAD-BEGIN")
                i1 = log.find(MARK + " PAYLOAD-END")
                print("CLAIMED run", run["id"], "conclusion", run.get("conclusion"), flush=True)
                if i0 >= 0 and i1 > i0:
                    print(log[i0:i1 + len(MARK) + 12], flush=True)
                else:
                    print(log[-6000:], flush=True)
                return
    print("RESULT: NOT-CLAIMED", flush=True)

if __name__ == "__main__":
    main()
