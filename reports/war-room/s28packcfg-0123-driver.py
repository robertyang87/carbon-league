"""S28PACKCFG0123 driver: ONE dispatch to spot-28 via cloud-assistant,
claim by unique marker only, per_page=30 + created_at truncation.
On 403 / rate-limit: stop immediately, no retry/polling.
Log retrieval via Range chunks (fetch_job_log_ranges) to beat ~15KB truncation.
"""
import sys, base64, time, urllib.error
sys.path.insert(0, "/home/hatch/workspace")
sys.path.insert(0, "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files")
import gh_api
from fetch_job_log_ranges import fetch_log

MARK = "S28PACKCFG0123"
BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
SCRIPT = BASE + "/s28packcfg-0123.sh"
LOG = BASE + "/s28packcfg-0123.log"
INST = "i-uf65egu9fey7xf7tu8ft"  # spot-28, per task verbatim
OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"

def is_403(e):
    return isinstance(e, urllib.error.HTTPError) and e.code in (403, 429)

script = open(SCRIPT).read()
assert MARK in script
since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 120))
print("claim_since (UTC, -120s skew buffer):", since, flush=True)
try:
    gh_api.dispatch(OWNER, REPO, WF, ref="main",
                    inputs={"instance_id": INST, "region": "cn-shanghai",
                            "command_b64": base64.b64encode(script.encode()).decode()})
except Exception as e:
    print("DISPATCH_FAILED:", repr(e), "code=", getattr(e, "code", None), flush=True)
    sys.exit(2)
print("dispatched", INST, "at", time.strftime("%H:%M:%S CST"), flush=True)

deadline = time.time() + 900
seen = set()
result = None
result_run = None
while time.time() < deadline:
    time.sleep(45)
    try:
        r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30")
    except Exception as e:
        print("LIST_ERROR:", repr(e), "code=", getattr(e, "code", None), flush=True)
        if is_403(e):
            print("RATE_LIMIT_STOP: 403/429 on list, stopping silently per owner rule", flush=True)
            sys.exit(3)
        continue
    for run in r.get("workflow_runs", [])[:30]:
        rid = run["id"]
        if rid in seen or run["status"] != "completed":
            continue
        if run.get("created_at", "") < since:
            continue
        seen.add(rid)
        try:
            jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs")
        except Exception as e:
            print("JOBS_ERROR run", rid, repr(e), flush=True)
            if is_403(e):
                print("RATE_LIMIT_STOP", flush=True); sys.exit(3)
            continue
        for j in jobs.get("jobs", []):
            try:
                log = fetch_log(j["id"])
            except Exception as e:
                print("LOG_ERROR job", j["id"], repr(e), flush=True)
                continue
            if isinstance(log, str) and MARK in log:
                print("CLAIMED run", rid, "conclusion", run.get("conclusion"), "created_at", run.get("created_at"), flush=True)
                open(LOG, "w").write(log)
                i0 = log.find(MARK + "_BEGIN"); i1 = log.find(MARK + "_END")
                seg = log[i0:i1 + len(MARK) + 4] if i0 >= 0 and i1 > i0 else log
                print(seg, flush=True)
                result = log; result_run = rid
                break
        if result:
            break
    if result:
        break
print("RESULT:", "CLAIMED run=" + str(result_run) if result else "NOT-CLAIMED", flush=True)
