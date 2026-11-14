#!/usr/bin/env python3
"""ppou2 serial driver (runs on my VM via nohup). Alternates m/c at round boundaries.

Schedule: first (c,4) so the control arm banks its first formal round (round-1
KL), then alternating checkpoint blocks (m +20 / c +40). State is derived from
L20 on every poll (stateless, restart-safe): the active entry is the first
schedule entry whose arm has not yet reached its target max_it.

Safety rules (hard):
- Never launch while any trainer is alive.
- Never SIGTERM the active arm before its target (needs state_match too).
- If a running arm's process disappears before its target and the driver did
  not stop it, ALERT and exit WITHOUT relaunching (relaunch is the war room's
  call, not the driver's).
- One launch retry, then ALERT and exit.
"""
import json, subprocess, sys, time

SSH_BASE = ["ssh", "-i", "/home/hatch/.ssh/id_ed25519", "-o", "IdentitiesOnly=yes",
            "-o", "UserKnownHostsFile=/home/hatch/.ssh/known_hosts",
            "-o", "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p",
            "root@100.79.251.87"]
LOG_PATH = "/home/hatch/workspace/carbon-live/ppou2/serial_driver.log"
POLL_S = 60

SCHEDULE = [("c", 4)]
_m, _c = 20, 40
while _m <= 200:
    SCHEDULE.append(("m", _m))
    SCHEDULE.append(("c", _c))
    _m += 20
    _c += 40


def log(msg):
    line = f"{time.strftime('%F %T %Z')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def ssh(cmd, timeout=90):
    r = subprocess.run(SSH_BASE + [cmd], capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def remote_state():
    rc, out, err = ssh("python3 /root/carbon/remote_serial_check.py")
    if rc != 0:
        raise RuntimeError(f"serial check failed rc={rc} err={err[:300]}")
    st = {}
    for line in out.splitlines():
        d = json.loads(line)
        st[d["arm"]] = d
    return st


def launch_record(arm):
    rc, out, _ = ssh(f"cat /root/carbon/data/rl3/ppou2_{arm}.launch.json")
    if rc != 0 or not out:
        return None
    try:
        return json.loads(out)
    except Exception:
        return None


def pid_alive(pid):
    _, out, _ = ssh(f"kill -0 {pid} && echo alive || echo dead")
    return "alive" in out


def gpu_pids():
    rc, out, _ = ssh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader")
    pids = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            pids.append(int(parts[0]))
    return pids


def stop_arm(arm):
    rec = launch_record(arm)
    if not rec:
        log(f"stop {arm}: no launch record, nothing to stop")
        return True
    pgid = rec["pgid"]
    log(f"stop {arm}: SIGTERM PGID {pgid} (pid {rec['pid']})")
    ssh(f"kill -TERM -{pgid}")
    for _ in range(24):  # up to 120s
        rc, out, _ = ssh(f"pgrep -g {pgid} || echo empty")
        if "empty" in out:
            break
        time.sleep(5)
    else:
        log(f"ALERT stop {arm}: PGID {pgid} still alive after 120s; NOT proceeding")
        return False
    time.sleep(5)
    left = gpu_pids()
    if left:
        log(f"stop {arm}: GPU still has pids {left}; killing strays by PID")
        for p in left:
            ssh(f"kill -TERM {p}")
        time.sleep(10)
        left = gpu_pids()
        if left:
            log(f"ALERT stop {arm}: GPU pids {left} survive SIGTERM; NOT proceeding")
            return False
    log(f"stop {arm}: group gone, GPU clear")
    return True


def launch_arm(arm, resume):
    cmd = f"/root/carbon/.venv/bin/python /root/carbon/launch_ppou2_v4.py {arm}"
    if resume:
        cmd += f" --resume /root/carbon/data/rl3/ppou2_{arm}/state_last.pt"
    log(f"launch {arm}{' (resume)' if resume else ' (fresh)'}")
    rc, out, err = ssh(cmd, timeout=120)
    if rc != 0:
        log(f"launch {arm} FAILED rc={rc} err={err[:300]}")
        return None
    try:
        rec = json.loads(out)
    except Exception:
        log(f"launch {arm}: unparsable launcher output: {out[:300]}")
        return None
    log(f"launch {arm}: pid {rec['pid']} pgid {rec['pgid']}")
    time.sleep(90)
    rc, out, _ = ssh(f"python3 /root/carbon/remote_verify_arm.py {arm} {rec['pid']}")
    try:
        v = json.loads(out)
    except Exception:
        log(f"verify {arm}: unparsable: {out[:300]}")
        return None
    ok = v["alive"] and v["gpu_mib"] > 0 and v["league_keys"] == 11 and v["spawn_main"] == 15
    log(f"verify {arm}: {v} -> {'OK' if ok else 'FAIL'}")
    return rec if ok else None


def main():
    log("serial driver START; schedule head: " + str(SCHEDULE[:6]))
    state = {"prev_alive": {}, "launch_failures": 0}
    while True:
        try:
            tick(state)
        except Exception as e:
            # any exception in a poll cycle is logged and retried; it must
            # never kill the driver (a dead driver strands the line)
            log(f"loop error: {type(e).__name__}: {e}; retry in {POLL_S}s")
        time.sleep(POLL_S)


def tick(state):
    st = remote_state()
    # heartbeat: one compact line per poll, real values (proves polling alive)
    hb = lambda a: f"{a} it={st[a]['max_it']} warm={st[a]['warm']} formal={st[a]['formal']}"
    log(f"poll {hb('m')} | {hb('c')}")
    prev_alive = state["prev_alive"]
    recs = {a: launch_record(a) for a in ("m", "c")}
    alive = {a: bool(recs[a] and pid_alive(recs[a]["pid"])) for a in ("m", "c")}
    # unexpected death: was alive last poll, gone now, driver did not stop it
    for a in ("m", "c"):
        if prev_alive.get(a) and not alive[a]:
            log(f"ALERT {a} process died before its target (not a driver stop); "
                f"max_it={st[a]['max_it']}; exiting without relaunch")
            sys.exit(3)
    state["prev_alive"] = dict(alive)
    # active entry: first schedule entry whose arm has not reached its target
    active_idx = next(i for i, (a, t) in enumerate(SCHEDULE) if st[a]["max_it"] < t)
    active_arm, target = SCHEDULE[active_idx]
    running = [a for a in ("m", "c") if alive[a]]
    if running and running != [active_arm]:
        for a in running:
            if not stop_arm(a):
                log("ALERT stop failed; exiting")
                sys.exit(4)
        running = []
        state["prev_alive"] = {"m": False, "c": False}
    if running == [active_arm]:
        s = st[active_arm]
        if s["max_it"] >= target:
            if not s["state_match"]:
                log(f"{active_arm} it={s['max_it']} >= {target} but state sha mismatch; "
                    f"waiting for state write")
            else:
                log(f"boundary reached: {active_arm} it={s['max_it']} >= {target}; switching")
                if not stop_arm(active_arm):
                    log("ALERT stop failed; exiting")
                    sys.exit(4)
                state["prev_alive"] = {"m": False, "c": False}
                next_arm = SCHEDULE[active_idx + 1][0]
                ok = launch_arm(next_arm, resume=(st[next_arm]["max_it"] > 0))
                if not ok:
                    state["launch_failures"] += 1
                    if state["launch_failures"] >= 2:
                        log("ALERT launch failed twice; exiting")
                        sys.exit(5)
                    log("launch failed once; retrying in 120s")
                    time.sleep(120)
                    return
                state["launch_failures"] = 0
                # verified alive at launch+90s: any death from here on trips
                # the unexpected-death check on the next poll
                state["prev_alive"] = {a: (a == next_arm) for a in ("m", "c")}
    elif not running:
        log(f"nothing running; active entry is ({active_arm},{target}); launching it")
        ok = launch_arm(active_arm, resume=(st[active_arm]["max_it"] > 0))
        if not ok:
            state["launch_failures"] += 1
            if state["launch_failures"] >= 2:
                log("ALERT launch failed twice; exiting")
                sys.exit(5)
            log("launch failed once; retrying in 120s")
            time.sleep(120)
            return
        state["launch_failures"] = 0
        state["prev_alive"] = {a: (a == active_arm) for a in ("m", "c")}


if __name__ == "__main__":
    main()
