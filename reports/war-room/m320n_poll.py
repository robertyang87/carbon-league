import subprocess, time, sys, os, base64

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = open(os.path.join(HERE, "m320n-probe-dgx.tpl")).read()
DRIVER = os.path.join(HERE, "m9xj260wr1_driver.py")

def wrap(inner, mark, tmo=300):
    b64 = base64.b64encode(inner.encode()).decode()
    return f"""#!/bin/bash
MARK={mark}
echo "${{MARK}}_BEGIN $(date '+%F %T %Z') host=$(hostname)"
echo '{b64}' | base64 -d > /tmp/m320n_inner_{mark}.sh
timeout -k 30 {tmo} ssh -i ~/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' < /tmp/m320n_inner_{mark}.sh 2>&1
echo "ssh-rc=$?"
echo "${{MARK}}_END"
"""

start = time.time()
DEADLINE = int(sys.argv[1]) if len(sys.argv) > 1 else 150 * 60
INTERVAL = 240
rnd = 0
launched = False
while time.time() - start < DEADLINE:
    rnd += 1
    mark = f"M320Q{rnd:02d}{time.strftime('%H%M')}"
    inner = TPL.replace("__MARK__", mark)
    spath = os.path.join(HERE, f"m320n_poll_r{rnd}.sh")
    opath = os.path.join(HERE, f"m320n_poll_r{rnd}.log")
    open(spath, "w").write(wrap(inner, mark))
    print(f"[poll] round {rnd} marker {mark} {time.strftime('%H:%M:%S')}", flush=True)
    r = subprocess.run([sys.executable, DRIVER, spath, mark, opath, "420"],
                       capture_output=True, text=True)
    tail = r.stdout[-1200:]
    print(tail, flush=True)
    if r.returncode != 0:
        print("[poll] driver rc:", r.returncode, r.stderr[-400:], flush=True)
    if os.path.exists(opath):
        log = open(opath, errors="replace").read()
        i0 = log.find(mark + " DGX")
        body = log[i0:] if i0 >= 0 else log
        if "NEEDS-RULING" in body:
            print("[poll] NEEDS-RULING detected, stopping", flush=True)
            print(body[:6000], flush=True)
            sys.exit(3)
        if "FAIL" in body and "CHAIN-BEGIN" in body and "FAIL " in body:
            print("[poll] CHAIN FAIL detected", flush=True)
            print(body[:6000], flush=True)
            sys.exit(2)
        if not launched and "CHAIN-DONE" in body:
            launched = True
            print("[poll] === LAUNCH THREE-PIECE (chain attestation) ===", flush=True)
            print(body[:8000], flush=True)
        # training completion: no trainer AND both logs show ep 19 done
        if launched and "NO-TRAINER" in body:
            import re
            eps = re.findall(r"== (r1|g1) eps=(\d+)", body)
            d = dict(eps)
            if d.get("r1") and d.get("g1") and int(d["r1"]) >= 20 and int(d["g1"]) >= 20:
                print("[poll] === TRAINING COMPLETE ===", flush=True)
                print(body[:8000], flush=True)
                sys.exit(0)
    if time.time() - start + INTERVAL >= DEADLINE:
        break
    time.sleep(INTERVAL)
print(f"[poll] TIMEOUT after {round(time.time()-start)}s rounds={rnd} launched={launched}", flush=True)
sys.exit(4)
