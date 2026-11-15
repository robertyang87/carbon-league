"""M7Bs v2 retrain driver: dispatch launcher, claim it, then poll probes
until the DGX stage script prints M7V2 DONE / FAIL. Progress appended to
m7v2-driver.log. Deadline ~150 min from dispatch (build + ~75 min train)."""
import gh_api, base64, time, os, sys

MARK = "M7V2K8R3"
BASE = os.path.expanduser("~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/")
LOGF = open(BASE + "m7v2-driver.log", "a", buffering=1)

def log(*a):
    s = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOGF.write(s + "\n")

def dispatch_script(script_text, note):
    gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
        inputs={"instance_id": "i-uf62k0o6zbe6j7zxnzg5", "region": "cn-shanghai",
                "command_b64": base64.b64encode(script_text.encode()).decode()})
    log("dispatched", note)

def claim_run(marker, timeout_s, since_iso=None, poll_s=20):
    """Poll recent cloud-assistant runs; return log text of first completed
    run containing marker, else None. per_page=15: other lines dispatch
    ~1/min, a 5-run window drops our run before it completes. since_iso
    (UTC ISO) filters out runs created before our dispatch."""
    deadline = time.time() + timeout_s
    seen = set()
    while time.time() < deadline:
        time.sleep(poll_s)
        try:
            r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=15")
        except Exception as e:
            log("list error:", e); continue
        for run in r.get("workflow_runs", [])[:15]:
            rid = run["id"]
            if rid in seen or run["status"] != "completed":
                continue
            if since_iso and run.get("created_at", "") < since_iso:
                continue
            seen.add(rid)
            try:
                jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
            except Exception as e:
                log("jobs error:", e); continue
            for j in jobs.get("jobs", []):
                try:
                    lg = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
                except Exception as e:
                    log("log error:", e); continue
                if isinstance(lg, str) and marker in lg:
                    log("CLAIMED run", rid, "job", j["id"], "conclusion", run.get("conclusion"))
                    return lg
    return None

def main():
    t0 = time.time()
    since_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 - 60))
    os.chdir(os.path.expanduser("~/workspace"))
    dgx_script = open(BASE + "m7v2-dgx.sh").read()
    wrapper = open(BASE + "m7v2.sh").read()
    assert wrapper.count("__DGX_B64__") == 1, "placeholder must occur exactly once"
    script = wrapper.replace("__DGX_B64__", base64.b64encode(dgx_script.encode()).decode(), 1)
    # Aliyun RunCommand content limit is 16KB base64; the 2026-11-12 16:58
    # dispatch failed with CmdContent.ExceedLimit because str.replace()
    # embedded the DGX script twice (placeholder also sat in a comment).
    assert len(script.encode()) < 14000, f"launcher too big: {len(script.encode())} bytes"
    log("launcher size bytes:", len(script.encode()))
    dispatch_script(script, "launcher " + MARK)

    lg = claim_run(MARK, 600, since_iso=since_iso)
    if not lg:
        log("LAUNCHER NOT-CLAIMED")
        return 1
    i0 = lg.find(MARK)
    log("LAUNCHER LOG (excerpt):")
    log(lg[max(0, i0 - 100):i0 + 3000])
    if "SCP_OK" not in lg:
        log("LAUNCHER FAIL: no SCP_OK in claimed log")
        return 1

    probe_tpl = open(BASE + "m7v2-probe.sh").read()
    assert "__PROBE_MARK__" in probe_tpl
    seq = 0
    while time.time() - t0 < 150 * 60:
        time.sleep(300)
        seq += 1
        pmark = "M7V2PRB%03d" % seq
        probe = probe_tpl.replace("__PROBE_MARK__", pmark)
        try:
            dispatch_script(probe, "probe " + pmark)
        except Exception as e:
            log("probe dispatch error:", e); continue
        plg = claim_run(pmark, 240, since_iso=since_iso)
        if not plg:
            log("probe", seq, "NOT-CLAIMED"); continue
        j0 = plg.find(pmark)
        chunk = plg[max(0, j0 - 100):j0 + 5000]
        open(BASE + "m7v2-probe-%03d.log" % seq, "w").write(plg)
        log("PROBE", seq, "output:")
        log(chunk)
        if "M7V2 DONE" in plg:
            log("COMPLETED: stage script printed M7V2 DONE")
            open(BASE + "m7v2-full.log", "w").write(plg)
            return 0
        if "M7V2 FAIL" in plg:
            log("FAILED: stage script printed M7V2 FAIL")
            open(BASE + "m7v2-full.log", "w").write(plg)
            return 1
    log("TIMEOUT 150min without DONE/FAIL")
    return 1

if __name__ == "__main__":
    rc = main()
    log("driver exit", rc)
    sys.exit(rc)
