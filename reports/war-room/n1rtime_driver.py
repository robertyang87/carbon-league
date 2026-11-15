#!/usr/bin/env python3
"""Generic cloud-assistant driver: dispatch a bash script (command_b64) to an
instance, poll workflow runs, claim ONLY the job log containing our marker."""
import base64, json, sys, time
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"

def main():
    script_path, marker, instance, timeout_s = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    script = open(script_path, "rb").read()
    b64 = base64.b64encode(script).decode()
    t0 = time.time()
    gh_api.dispatch(OWNER, REPO, WF, ref="main",
                    inputs={"instance_id": instance, "region": "cn-shanghai",
                            "command": "", "command_b64": b64})
    print(f"dispatched marker={marker} instance={instance}", flush=True)
    seen_runs = set()
    while time.time() - t0 < timeout_s:
        time.sleep(20)
        runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=15")
        for run in runs.get("workflow_runs", []):
            rid = run["id"]
            if rid in seen_runs or run["status"] != "completed":
                continue
            seen_runs.add(rid)
            jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs")
            for job in jobs.get("jobs", []):
                try:
                    logs = gh_api.get_job_logs(OWNER, REPO, job["id"])
                except Exception as e:
                    print(f"run {rid} job {job['id']} logs failed: {e}", flush=True)
                    continue
                if marker in logs:
                    print(f"CLAIMED run={rid} job={job['id']} conclusion={run.get('conclusion')}")
                    # print only the region between first and last marker occurrence
                    lines = logs.splitlines()
                    idx = [i for i, ln in enumerate(lines) if marker in ln]
                    lo, hi = idx[0], idx[-1]
                    print("\n".join(lines[max(0, lo-1):hi+2]))
                    return
        print(f"...polling {int(time.time()-t0)}s, runs seen={len(seen_runs)}", flush=True)
    print("NOT-CLAIMED within timeout")

if __name__ == "__main__":
    main()
