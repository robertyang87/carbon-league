"""Claim S28PACKCFG0123 run with ONE list call + curl log fetch.
Window anchored at the single dispatch (2026-11-12T17:24:16Z approx).
Stops immediately on 403/429."""
import sys, os
sys.path.insert(0, "/home/hatch/workspace")
sys.path.insert(0, "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files")
import gh_api
from fetch_log_curl import fetch as curl_fetch

MARK = "S28PACKCFG0123"
BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"
LO, HI = "2026-11-12T17:23:00Z", "2026-11-12T17:35:00Z"

try:
    r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30")
except Exception as e:
    print("LIST_ERROR", repr(e), "code=", getattr(e, "code", None), flush=True)
    sys.exit(3)
cands = []
for run in r.get("workflow_runs", []):
    ca = run.get("created_at", "")
    print("RUN", run["id"], ca, run["status"], run.get("conclusion"), flush=True)
    if LO <= ca <= HI and run["status"] == "completed":
        cands.append(run)
print("candidates in window:", [c["id"] for c in cands], flush=True)
for run in cands:
    rid = run["id"]
    try:
        jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs")
    except Exception as e:
        print("JOBS_ERROR", rid, repr(e), flush=True)
        if getattr(e, "code", None) in (403, 429):
            print("RATE_LIMIT_STOP", flush=True); sys.exit(3)
        continue
    for j in jobs.get("jobs", []):
        out = f"{BASE}/s28packcfg-0123-run{rid}.log"
        got = curl_fetch(j["id"], out, tries=2)
        if not got:
            print("CURL_FAIL run", rid, "job", j["id"], flush=True)
            continue
        txt = open(out, errors="replace").read()
        print("run", rid, "job", j["id"], "log chars", len(txt), "marker:", MARK in txt, flush=True)
        if MARK in txt:
            i0 = txt.find(MARK + "_BEGIN"); i1 = txt.find(MARK + "_END")
            seg = txt[i0:i1 + len(MARK) + 4] if i0 >= 0 and i1 > i0 else txt
            open(BASE + "/s28packcfg-0123.log", "w").write(txt)
            print("CLAIMED run", rid, "conclusion", run.get("conclusion"), flush=True)
            print(seg, flush=True)
            sys.exit(0)
print("NOT-CLAIMED in window", flush=True)
