import subprocess, time, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = open(os.path.join(HERE, "m320_poll_tpl.sh")).read()
DRIVER = os.path.join(HERE, "m9xj260wr1_driver.py")

start = time.time()
DEADLINE = 85 * 60          # ~85 min cap (task allows ~90)
INTERVAL = 240              # ~4 min between polls
rnd = 0
last_log = None

while time.time() - start < DEADLINE:
    rnd += 1
    hhmm = time.strftime("%H%M")
    mark = f"M320P{hhmm}{chr(64 + min(rnd, 26))}"
    script = TPL.replace("__MARK__", mark)
    spath = os.path.join(HERE, f"m320_poll_r{rnd}.sh")
    opath = os.path.join(HERE, f"m320_poll_r{rnd}.log")
    open(spath, "w").write(script)
    print(f"[waiter] round {rnd} marker {mark} dispatching {time.strftime('%H:%M:%S')}", flush=True)
    r = subprocess.run([sys.executable, DRIVER, spath, mark, opath, "600"],
                       capture_output=True, text=True)
    print(r.stdout[-1500:], flush=True)
    if r.returncode != 0:
        print("[waiter] driver rc != 0:", r.returncode, r.stderr[-500:], flush=True)
    if os.path.exists(opath):
        last_log = open(opath, errors="replace").read()
        if "READY320=YES" in last_log:
            print(f"[waiter] READY detected round {rnd} after {round(time.time()-start)}s", flush=True)
            i0 = last_log.find(mark + " DGX")
            print(last_log[i0 if i0 >= 0 else 0:], flush=True)
            sys.exit(0)
        if "CHAIN320FAIL" in last_log:
            print("[waiter] chain FAIL signal", flush=True)
            sys.exit(2)
    if time.time() - start + INTERVAL >= DEADLINE:
        break
    time.sleep(INTERVAL)

print(f"[waiter] TIMEOUT after {round(time.time()-start)}s, rounds={rnd}", flush=True)
if last_log:
    i0 = last_log.find(" DGX ")
    print(last_log[max(0, i0 - 40):] if i0 >= 0 else last_log[-4000:], flush=True)
sys.exit(3)
