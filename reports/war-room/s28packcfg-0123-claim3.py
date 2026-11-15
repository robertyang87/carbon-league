"""Claim page 2/3: our run was created ~2026-11-12T17:24:16Z, now beyond top-30.
Two list calls max (page=2, page=3), curl-check only runs created 17:24:00-17:26:30Z.
Stop on 403/429."""
import sys
sys.path.insert(0, "/home/hatch/workspace")
sys.path.insert(0, "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files")
import gh_api
from fetch_log_curl import fetch as curl_fetch

MARK = "S28PACKCFG0123"
BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"
LO, HI = "2026-11-12T17:24:00Z", "2026-11-12T17:26:30Z"

cands = []
for page in (2, 3):
    try:
        r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30&page={page}")
    except Exception as e:
        print("LIST_ERROR page", page, repr(e), "code=", getattr(e, "code", None), flush=True)
        sys.exit(3)
    runs = r.get("workflow_runs", [])
    print(f"page{page}: {len(runs)} runs, range {runs[-1]['created_at'] if runs else '-'} .. {runs[0]['created_at'] if runs else '-'}", flush=True)
    for run in runs:
        ca = run.get("created_at", "")
        if LO <= ca <= HI and run["status"] == "completed":
            cands.append(run)
            print("CAND", run["id"], ca, run.get("conclusion"), flush=True)
    if runs and runs[-1]["created_at"] < LO:
        break
print("candidates:", [c["id"] for c in cands], flush=True)
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
            print("CURL_FAIL run", rid, "job", j["id"], flush=True); continue
        txt = open(out, errors="replace").read()
        print("run", rid, "job", j["id"], "chars", len(txt), "marker:", MARK in txt, flush=True)
        if MARK in txt:
            open(BASE + "/s28packcfg-0123.log", "w").write(txt)
            print("CLAIMED run", rid, "conclusion", run.get("conclusion"), flush=True)
            i0 = txt.find(MARK + "_BEGIN"); i1 = txt.find(MARK + "_END")
            print(txt[i0:i1 + len(MARK) + 4] if i0 >= 0 and i1 > i0 else txt, flush=True)
            sys.exit(0)
print("NOT-CLAIMED", flush=True)
