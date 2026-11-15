"""H24 read-only probe driver, run 2026-11-12 ~23:59 CST. Marker: H24PROBE2359.
Target: spot-28 (i-uf65egu9fey7xf7tu8ft), cn-shanghai. READ-ONLY.
Claim by marker; log fetch keeps partial bytes on truncation (tonight's
Actions-log downloads truncate ~15KB, so fall back to partial capture).
Usage: PYTHONPATH=/home/hatch/workspace python3 h24_probe_20261112_2359-driver.py
"""
import sys, base64, time, urllib.request, urllib.error, http.client
sys.path.insert(0, "/home/hatch/workspace")
sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
import gh_api
from dynamic_credentials import add_surrogate_to_request

MARK = "H24PROBE2359"
HF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
SCRIPT = HF + "/h24_probe_20261112_2359.sh"
LOG = HF + "/h24_probe_20261112_2359.log"
INST = "i-uf65egu9fey7xf7tu8ft"  # spot-28

def fetch_log(job_id):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    url = f"https://api.github.com/repos/robertyang87/carbon-league/actions/jobs/{job_id}/logs"
    r = urllib.request.Request(url, method="GET", headers={"User-Agent": "muse-agent"})
    add_surrogate_to_request(r, "custom.github", allowed_hosts=["api.github.com"])
    loc = None
    try:
        with opener.open(r, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location")
        if not loc:
            raise
    r2 = urllib.request.Request(loc, method="GET", headers={"User-Agent": "muse-agent"})
    try:
        with urllib.request.urlopen(r2, timeout=60) as resp2:
            return resp2.read().decode("utf-8", errors="replace")
    except http.client.IncompleteRead as ir:
        return (ir.partial or b"").decode("utf-8", errors="replace")

script = open(SCRIPT).read()
assert MARK in script, "marker not in script"
t0 = time.time()
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": INST, "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched", INST, "at", time.strftime("%H:%M:%S"), flush=True)

deadline = time.time() + 420
seen = set()
result = None
while time.time() < deadline:
    time.sleep(15)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
    except Exception as e:
        print("list error:", e, flush=True)
        continue
    for run in r.get("workflow_runs", [])[:30]:
        rid = run["id"]
        if rid in seen or run["status"] != "completed":
            continue
        seen.add(rid)
        try:
            jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
        except Exception:
            continue
        for j in jobs.get("jobs", []):
            try:
                log = fetch_log(j["id"])
            except Exception as e:
                print("log fetch error:", e, flush=True)
                continue
            if isinstance(log, str) and MARK in log:
                print("CLAIMED run", rid, "conclusion", run.get("conclusion"), flush=True)
                open(LOG, "w").write(log)
                i0 = log.find(MARK + "_BEGIN")
                i1 = log.find(MARK + "_END")
                print(log[i0:i1 + len(MARK) + 4] if i0 >= 0 else log[:8000], flush=True)
                result = log
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time() - t0), "CLAIMED" if result else "NOT-CLAIMED", flush=True)
