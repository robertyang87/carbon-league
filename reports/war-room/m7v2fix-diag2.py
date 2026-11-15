"""Dispatch + claim the DGX diagnostic probe (diag2, ssh-wrapped)."""
import gh_api, base64, time, os, sys

BASE = os.path.expanduser("~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/")
PMARK = "M7V2FXDIAG02"
t0 = time.time()
since_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 - 60))
os.chdir(os.path.expanduser("~/workspace"))

diag = open(BASE + "m7v2fix-diag2.sh").read().replace("__DIAG_MARK__", PMARK)
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": "i-uf62k0o6zbe6j7zxnzg5", "region": "cn-shanghai",
            "command_b64": base64.b64encode(diag.encode()).decode()})
print("dispatched diag", PMARK, flush=True)

seen = set()
deadline = time.time() + 300
while time.time() < deadline:
    time.sleep(20)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=15")
    except Exception as e:
        print("list error:", e, flush=True); continue
    for run in r.get("workflow_runs", [])[:15]:
        rid = run["id"]
        if rid in seen or run["status"] != "completed":
            continue
        if run.get("created_at", "") < since_iso:
            continue
        seen.add(rid)
        try:
            jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
        except Exception as e:
            print("jobs error:", e, flush=True); continue
        for j in jobs.get("jobs", []):
            try:
                lg = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
            except Exception as e:
                print("log error:", e, flush=True); continue
            if isinstance(lg, str) and PMARK in lg:
                print("CLAIMED run", rid, "job", j["id"], "conclusion", run.get("conclusion"), flush=True)
                open(BASE + "m7v2fix-diag-002.log", "w").write(lg)
                j0 = lg.find(PMARK)
                print(lg[max(0, j0 - 100):j0 + 8000], flush=True)
                sys.exit(0)
print("DIAG2 NOT-CLAIMED within 300s", flush=True)
sys.exit(1)
