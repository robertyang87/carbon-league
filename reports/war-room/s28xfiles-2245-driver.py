import sys, base64, time
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"
INSTANCE = "i-uf62k0o6zbe6j7zxnzg5"  # L20 gpu-l20-5
seen_runs = set()

def dispatch(script_path):
    script = open(script_path).read()
    gh_api.dispatch(OWNER, REPO, WF, ref="main",
                    inputs={"instance_id": INSTANCE, "region": "cn-shanghai",
                            "command_b64": base64.b64encode(script.encode()).decode()})
    print("dispatched", script_path, "at", time.strftime("%H:%M:%S"), flush=True)

def claim(marker, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(20)
        try:
            r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30")
        except Exception as e:
            print("list error:", e, flush=True); continue
        for run in r.get("workflow_runs", [])[:30]:
            rid = run["id"]
            if rid in seen_runs or run["status"] != "completed":
                continue
            seen_runs.add(rid)
            try:
                jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs")
            except Exception as e:
                print("jobs error:", e, flush=True); continue
            for j in jobs.get("jobs", []):
                try:
                    log = gh_api.get_job_logs(OWNER, REPO, j["id"])
                except Exception as e:
                    print("log error:", e, flush=True); continue
                if isinstance(log, str) and marker in log:
                    print("CLAIMED run", rid, "conclusion", run.get("conclusion"), flush=True)
                    i0 = log.find(marker)
                    return log[max(0, i0 - 200):i0 + 20000]
    return None

def main():
    dispatch(f"{BASE}/s28xfiles-2245.sh")
    log = claim("S28XFILES2245", 600)
    if log is None:
        print("XFILES-NOT-CLAIMED", flush=True); return
    with open(f"{BASE}/s28xfiles-2245.log", "w") as f:
        f.write(log)
    print(log, flush=True)
    print("RESULT: DONE", flush=True)

if __name__ == "__main__":
    main()
