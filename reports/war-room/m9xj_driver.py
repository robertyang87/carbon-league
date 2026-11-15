import sys, os, base64, time
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

HF = os.path.expanduser("~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files")
INSTANCE_ID = "i-uf62k0o6zbe6j7zxnzg5"  # L20 gpu-l20-5, DGX relay host (verified in prior drivers)
assert INSTANCE_ID == "i-uf62k0o6zbe6j7zxnzg5"

def build_launch_script():
    chain = open(os.path.join(HF, "m9xj-chain-dgx.sh")).read()
    wrap = open(os.path.join(HF, "m9xj-launch-l20.sh")).read()
    assert "__CHAIN_BODY__" in wrap
    assert "CHAINSCRIPT" not in chain and "\nOUTER" not in chain
    return wrap.replace("__CHAIN_BODY__", chain.rstrip("\n"))

def dispatch(script):
    gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
        inputs={"instance_id": INSTANCE_ID, "region": "cn-shanghai",
                "command_b64": base64.b64encode(script.encode()).decode()})

def claim(mark, t0, deadline_s=600, window=20000):
    seen = set(); result = None
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        time.sleep(15)
        try:
            r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
        except Exception as e:
            print("list error:", e, flush=True); continue
        for run in r.get("workflow_runs", []):
            rid = run["id"]
            if rid in seen or run["status"] != "completed":
                continue
            # only consider runs created around/after dispatch
            seen.add(rid)
            try:
                jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
            except Exception as e:
                print("jobs error:", e, flush=True); continue
            for j in jobs.get("jobs", []):
                try:
                    log = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
                except Exception as e:
                    print("log error:", e, flush=True); continue
                if isinstance(log, str) and mark in log:
                    print("CLAIMED run", rid, "job", j["id"], "conclusion", run.get("conclusion"), flush=True)
                    i0 = log.find(mark)
                    print(log[max(0, i0-200):i0+window], flush=True)
                    result = log
                    break
            if result: break
        if result: break
    print("elapsed", round(time.time()-t0), "claimed" if result else "NOT-CLAIMED", flush=True)
    return result

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "launch":
        script = build_launch_script()
        print("launch script bytes:", len(script), flush=True)
        t0 = time.time()
        dispatch(script)
        print("dispatched at", time.strftime("%H:%M:%S"), flush=True)
        claim("M9XJCHN2026A", t0, deadline_s=600)
    elif mode == "pushfix":
        script = open(os.path.join(HF, "m9xj-pushfix-l20.sh")).read()
        t0 = time.time()
        dispatch(script)
        print("pushfix dispatched at", time.strftime("%H:%M:%S"), flush=True)
        claim("M9XJPUSH2050B", t0, deadline_s=900, window=12000)
    elif mode == "lanpush":
        script = open(os.path.join(HF, "m9xj-lanpush-l20.sh")).read()
        t0 = time.time()
        dispatch(script)
        print("lanpush dispatched at", time.strftime("%H:%M:%S"), flush=True)
        claim("M9XJLAN2103C", t0, deadline_s=900, window=16000)
    elif mode == "keypush":
        script = open(os.path.join(HF, "m9xj-keypush-l20.sh")).read()
        t0 = time.time()
        dispatch(script)
        print("keypush dispatched at", time.strftime("%H:%M:%S"), flush=True)
        claim("M9XJKEY2110D", t0, deadline_s=900, window=14000)
    elif mode == "probe":
        mark = sys.argv[2]
        probe = open(os.path.join(HF, "m9xj-probe-l20.sh")).read().replace("M9XJPRB_PLACEHOLDER", mark)
        t0 = time.time()
        dispatch(probe)
        print("probe dispatched at", time.strftime("%H:%M:%S"), "mark", mark, flush=True)
        claim(mark, t0, deadline_s=600, window=30000)
