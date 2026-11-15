"""Hakimi v2 pro TRIP 4 driver: ONE dispatch + claim per invocation.

Fixes vs trip-2 driver:
- Candidate runs (created after this dispatch) are RE-CHECKED every poll
  until completed; trip-2 marked a run 'seen' while in_progress and never
  looked at it again, missing its own run entirely.
- Log fetch: tolerant full fetch xN (keep longest) + Range-chunk stitched
  fetch as fallback when the plain fetch truncates (~15,364B cut).
- On HTTP 403/429 (shared API pool): silent long sleep, no busy polling;
  rate-limit sleep does not consume the claim deadline.
Usage: hakimi2-driver3.py SCRIPT MARKER DEADLINE_S DELAY_S CLAIM_OUT
"""
import sys, os, base64, time
sys.path.insert(0, "/home/hatch/workspace")
import gh_api, urllib.error, urllib.request
from datetime import datetime, timezone

SCRIPT, MARK = sys.argv[1], sys.argv[2]
DEADLINE_S, DELAY_S, CLAIM_OUT = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
HF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"
INSTANCE, REGION = "i-uf62k0o6zbe6j7zxnzg5", "cn-shanghai"

script = open(SCRIPT).read()
print(f"script {SCRIPT} bytes: {len(script)} b64 bytes: {len(base64.b64encode(script.encode()))} marker: {MARK}", flush=True)

if DELAY_S > 0:
    print(f"delaying {DELAY_S}s before dispatch", flush=True)
    t_end = time.time() + DELAY_S
    while time.time() < t_end:
        time.sleep(min(60, t_end - time.time()))
        print(f"delay remaining {int(t_end - time.time())}s", flush=True)

try:
    rl = gh_api.req("GET", "/rate_limit")
    print("rate_limit core:", rl["resources"]["core"], flush=True)
except Exception as e:
    print("rate_limit check failed:", e, flush=True)

def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def is_rl(e):
    return isinstance(e, urllib.error.HTTPError) and e.code in (403, 429)

t_dispatch = datetime.now(timezone.utc)
t_dispatch_s = t_dispatch.strftime("%H:%M:%S")
try:
    resp = gh_api.dispatch(OWNER, REPO, WF, ref="main",
        inputs={"instance_id": INSTANCE, "region": REGION,
                "command_b64": base64.b64encode(script.encode()).decode()})
except Exception as e:
    print("DISPATCH-ERROR:", repr(e), flush=True)
    sys.exit(3)
print("dispatched at", t_dispatch_s, "UTC resp:", resp, flush=True)

def _loc_for(job_id):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    url = f"{gh_api.API}/repos/{OWNER}/{REPO}/actions/jobs/{job_id}/logs"
    r = urllib.request.Request(url, method="GET", headers={"User-Agent": "muse-agent"})
    gh_api.add_surrogate_to_request(r, "custom.github", allowed_hosts=["api.github.com"])
    try:
        with opener.open(r, timeout=30) as resp:
            return None, resp.read()
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location")
        if not loc:
            raise
        return loc, None

def fetch_log_tolerant(job_id):
    """Return (text, nbytes, how). Plain fetch following the 302 manually."""
    try:
        loc, data = _loc_for(job_id)
        if data is not None:
            return data.decode("utf-8", errors="replace"), len(data), "full-direct"
    except Exception as e:
        return "", 0, f"err:{type(e).__name__}"
    try:
        r2 = urllib.request.Request(loc, method="GET", headers={"User-Agent": "muse-agent"})
        with urllib.request.urlopen(r2, timeout=60) as resp2:
            data = resp2.read()
            return data.decode("utf-8", errors="replace"), len(data), "full"
    except Exception as e:
        partial = getattr(e, "partial", b"") or b""
        return partial.decode("utf-8", errors="replace"), len(partial), f"partial:{type(e).__name__}"

def fetch_log_stitched(job_id, chunk=14000):
    """Range-chunk retrieval of the whole blob (DGX-line remedy)."""
    loc, data = _loc_for(job_id)
    if data is not None:
        return data.decode("utf-8", errors="replace"), len(data), "direct"
    # learn total size
    r = urllib.request.Request(loc, method="GET", headers={"User-Agent": "muse-agent", "Range": "bytes=0-0"})
    with urllib.request.urlopen(r, timeout=30) as resp:
        cr = resp.headers.get("Content-Range", "")
        total = int(cr.rsplit("/", 1)[1]) if "/" in cr else 0
    if total <= 0:
        return "", 0, "no-total"
    parts, start = [], 0
    while start < total:
        end = min(start + chunk - 1, total - 1)
        got = None
        for _ in range(3):
            try:
                rr = urllib.request.Request(loc, method="GET", headers={"User-Agent": "muse-agent", "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(rr, timeout=30) as resp:
                    got = resp.read()
                if len(got) == end - start + 1:
                    break
                got = None
            except Exception:
                got = None
            time.sleep(3)
        if got is None:
            return b"".join(parts).decode("utf-8", errors="replace"), sum(len(p) for p in parts), f"stitched-partial@{start}"
        parts.append(got)
        start = end + 1
    blob = b"".join(parts)
    return blob.decode("utf-8", errors="replace"), len(blob), "stitched"

candidates = {}   # rid -> {"created": ts, "status": str, "log_checks": int, "done": bool}
rl_sleep_total = 0
deadline = time.time() + DEADLINE_S
result = None
while time.time() < deadline + rl_sleep_total:
    time.sleep(20)
    try:
        r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30")
    except Exception as e:
        if is_rl(e):
            print("rate-limited listing runs; silent wait 120s", flush=True)
            rl_sleep_total += 120; time.sleep(120); continue
        print("list error:", repr(e), flush=True); continue
    for run in r.get("workflow_runs", []):
        rid = run["id"]
        try:
            created = parse_ts(run["created_at"])
        except Exception:
            continue
        if created < t_dispatch and (t_dispatch - created).total_seconds() > 90:
            continue
        c = candidates.setdefault(rid, {"created": created, "log_checks": 0, "done": False})
        if run["status"] == "completed" and not c["done"]:
            c["status"] = "completed"; c["conclusion"] = run.get("conclusion")
        else:
            c["status"] = run["status"]
    # check completed candidates, earliest first, within +240s of dispatch
    for rid in sorted(candidates, key=lambda k: candidates[k]["created"]):
        c = candidates[rid]
        if c.get("status") != "completed" or c["done"]:
            continue
        if (c["created"] - t_dispatch).total_seconds() > 240:
            c["done"] = True; continue
        try:
            jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs")
        except Exception as e:
            if is_rl(e):
                print("rate-limited listing jobs; silent wait 120s", flush=True)
                rl_sleep_total += 120; time.sleep(120)
            else:
                print("jobs error:", repr(e), flush=True)
            continue
        for j in jobs.get("jobs", []):
            best, how = "", ""
            for attempt in range(4):
                log, n, h = fetch_log_tolerant(j["id"])
                if len(log) > len(best):
                    best, how = log, h
                print(rid, j["id"], "attempt", attempt, "bytes", n, h, "marker" if MARK in log else "no-marker", flush=True)
                if MARK in log:
                    break
                time.sleep(3)
            if MARK not in best:
                log2, n2, h2 = fetch_log_stitched(j["id"])
                print(rid, j["id"], "stitched bytes", n2, h2, "marker" if MARK in log2 else "no-marker", flush=True)
                if len(log2) > len(best):
                    best, how = log2, h2
            c["log_checks"] += 1
            if MARK in best:
                print("CLAIMED run", rid, "job", j["id"], "conclusion", c.get("conclusion"), flush=True)
                i0 = best.find(MARK)
                print(best[max(0, i0 - 200):i0 + 14000], flush=True)
                with open(CLAIM_OUT, "w") as f:
                    f.write(f"run={rid} job={j['id']} conclusion={c.get('conclusion')} dispatched_at={t_dispatch_s} how={how}\n" + best)
                print("saved claim log", CLAIM_OUT, flush=True)
                result = (rid, j["id"], c.get("conclusion"))
                break
            if c["log_checks"] >= 2:
                c["done"] = True
        if result:
            break
    if result:
        break
    pend = {k: v["status"] for k, v in candidates.items() if not v["done"]}
    print(f"poll: candidates={len(candidates)} pending={pend}", flush=True)

if result:
    print("RESULT CLAIMED", result, flush=True)
    sys.exit(0)
print("RESULT NOT-CLAIMED candidates:", {k: (v["status"], v.get("conclusion"), v["log_checks"]) for k, v in candidates.items()}, flush=True)
sys.exit(2)
