#!/usr/bin/env python3
"""ppou2 serial driver (runs on my VM via nohup). Alternates m/c at round boundaries.

Schedule: first (c,4) so the control arm banks its first formal round (round-1
KL), then alternating checkpoint blocks (m +20 / c +40). State is derived from
L20 on every poll (stateless, restart-safe): the active entry is the first
schedule entry whose arm has not yet reached its target max_it.

KL stop-line mechanism (pre-registration + strategy-group ruling 2026-11-15
10:15): kl_current persists in kl_state.json (starts 0.05). Any formal round
with achieved KL >= 0.13 while kl_current is 0.05: stop the arm first, set
kl_current to 0.15 (both arms; state files edited as belt-and-braces via
remote_edit_kl.py before resume), resume toward the same target. If the
trigger round is the block target, the normal boundary switch carries the new
coefficient. A >= 0.13 round while kl_current is already 0.15: stop the arm
and set the HOLD flag (kl_hold.flag) -- no further automatic step; a human /
strategy-group ruling is required. While HOLD is set the driver polls and
logs but never launches. The first formal round at 0.15 is logged as a
KL15_CALIBRATION line for same-round reporting.

Safety rules (hard):
- Never launch while any trainer is alive.
- Never SIGTERM the active arm before its target (needs state_match too),
  except the KL stop-line above, which is itself pre-registered.
- If a running arm's process disappears before its target and the driver did
  not stop it, ALERT and exit WITHOUT relaunching (relaunch is the war room's
  call, not the driver's).
- One launch retry, then ALERT and exit.
"""
import json, os, subprocess, sys, time

SSH_BASE = ["ssh", "-i", "/home/hatch/.ssh/id_ed25519", "-o", "IdentitiesOnly=yes",
            "-o", "UserKnownHostsFile=/home/hatch/.ssh/known_hosts",
            "-o", "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p",
            "root@100.79.251.87"]
BASE_DIR = "/home/hatch/workspace/carbon-live/ppou2"
LOG_PATH = BASE_DIR + "/serial_driver.log"
KL_STATE_PATH = BASE_DIR + "/kl_state.json"
KL_HOLD_PATH = BASE_DIR + "/kl_hold.flag"
POLL_S = 60
KL_STOP = 0.13
KL_STEPS = (0.05, 0.15)

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


def remote_kl():
    rc, out, err = ssh("python3 /root/carbon/remote_kl_check.py")
    if rc != 0:
        raise RuntimeError(f"kl check failed rc={rc} err={err[:300]}")
    kl = {}
    for line in out.splitlines():
        d = json.loads(line)
        kl[d["arm"]] = d
    return kl


def kl_load():
    try:
        with open(KL_STATE_PATH) as f:
            d = json.load(f)
        return {"current": float(d["current"]),
                "calib_pending": bool(d.get("calib_pending", False)),
                "last_seen_it": dict(d.get("last_seen_it", {}))}
    except Exception:
        return {"current": KL_STEPS[0], "calib_pending": False,
                "last_seen_it": {}}


def kl_save(d):
    with open(KL_STATE_PATH, "w") as f:
        json.dump(d, f)


def kl_hold():
    return os.path.exists(KL_HOLD_PATH)


def state_file_kl(arm):
    rc, out, _ = ssh(
        "/root/carbon/.venv/bin/python -c \""
        "import torch, json; "
        f"st = torch.load('/root/carbon/data/rl3/ppou2_{arm}/state_last.pt', "
        "map_location='cpu', weights_only=False); "
        "print(json.dumps({'kl_coef': st.get('kl_coef'), 'it': st.get('it')}))\"")
    if rc != 0 or not out:
        return None
    try:
        return json.loads(out)
    except Exception:
        return None


def ensure_state_kl(arm, target):
    """Belt-and-braces: the arm's state file must carry kl_current before a
    resume (R03 makes --kl win anyway; the ruling asks for both paths)."""
    cur = state_file_kl(arm)
    if cur is None:
        log(f"ensure_state_kl {arm}: state unreadable; relying on --kl only")
        return
    if cur.get("kl_coef") == target:
        return
    state_path = f"/root/carbon/data/rl3/ppou2_{arm}/state_last.pt"
    bak = f"{state_path}.kl{cur.get('kl_coef')}.bak"
    rc, out, err = ssh(
        f"/root/carbon/.venv/bin/python /root/carbon/remote_edit_kl.py "
        f"{state_path} {target} {bak}", timeout=180)
    ok = rc == 0 and "EDIT_OK" in out
    log(f"ensure_state_kl {arm}: {cur.get('kl_coef')} -> {target} "
        f"{'OK' if ok else 'FAILED rc=' + str(rc) + ' ' + (out + err)[:300]}")
    if not ok:
        raise RuntimeError(f"state kl edit failed for {arm}")


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
    kl = kl_load()
    if resume:
        ensure_state_kl(arm, kl["current"])
    cmd = (f"/root/carbon/.venv/bin/python /root/carbon/launch_ppou2_v5.py {arm}"
           f" --kl-value {kl['current']}")
    if resume:
        cmd += f" --resume /root/carbon/data/rl3/ppou2_{arm}/state_last.pt"
    log(f"launch {arm}{' (resume)' if resume else ' (fresh)'} "
        f"kl_coef={kl['current']}")
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
    state = {"prev_alive": {}, "launch_failures": 0, "kl_trigger": {}}
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
    # ---- KL stop-line bookkeeping (strategy-group ruling 2026-11-15 10:15)
    kl = kl_load()
    try:
        kls = remote_kl()
    except Exception as e:
        log(f"kl check failed: {e}; skipping KL logic this poll")
        kls = None
    if kls is not None:
        for a in ("m", "c"):
            rec = kls[a]
            if rec["it"] is None:
                continue
            seen = kl["last_seen_it"].get(a, -1)
            if rec["it"] <= seen:
                continue
            kl["last_seen_it"][a] = rec["it"]
            if kl["calib_pending"] and rec["kl"] is not None:
                log(f"KL15_CALIBRATION {a} it={rec['it']} kl={rec['kl']} "
                    f"kl_coef={rec['kl_coef']} (first formal round at 0.15; "
                    f"same-round report required)")
                if rec["kl"] >= 0.10:
                    log(f"KL15_CALIBRATION_HIGH {a} it={rec['it']} "
                        f"kl={rec['kl']} >= 0.10 at kl_coef 0.15; "
                        f"strategy group must rule the next step")
                kl["calib_pending"] = False
            if (rec["kl"] is not None and rec["kl"] >= KL_STOP
                    and state.get("kl_handled_it", {}).get(a) != rec["it"]):
                state.setdefault("kl_trigger", {})[a] = rec
                log(f"KL_TRIGGER {a} it={rec['it']} kl={rec['kl']} >= "
                    f"{KL_STOP} at kl_coef={kl['current']}")
        kl_save(kl)
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
    # ---- KL HOLD: a second >= 0.13 trigger parks the line for a ruling
    if kl_hold():
        running_now = [a for a in ("m", "c") if alive[a]]
        for a in running_now:
            if not stop_arm(a):
                log("ALERT stop failed while enforcing KL HOLD; exiting")
                sys.exit(4)
        if running_now:
            state["prev_alive"] = {"m": False, "c": False}
        log("KL HOLD set; line parked, no launch (awaiting ruling)")
        return
    # active entry: first schedule entry whose arm has not reached its target
    active_idx = next(i for i, (a, t) in enumerate(SCHEDULE) if st[a]["max_it"] < t)
    active_arm, target = SCHEDULE[active_idx]
    running = [a for a in ("m", "c") if alive[a]]
    # ---- KL trigger on the running arm: stop first (pre-registered)
    trig = state.get("kl_trigger", {})
    if running and running[0] in trig:
        a = running[0]
        rec = trig.pop(a)
        kl = kl_load()
        if kl["current"] < KL_STEPS[-1]:
            old = kl["current"]
            kl["current"] = KL_STEPS[-1]
            kl["calib_pending"] = True
            # snapshot current its: only rounds landing AFTER this step count
            # as new (prevents re-reading the trigger round as calibration)
            try:
                snap = {r["arm"]: r["it"] for r in remote_kl().values()}
                if snap:
                    kl["last_seen_it"] = snap
            except Exception as e:
                log(f"kl snapshot after step failed: {e!r}")
            kl_save(kl)
            log(f"KL_ACTION {a} it={rec['it']} kl={rec['kl']}: stop first, "
                f"kl_coef {old} -> {kl['current']} (both arms); "
                f"{'boundary round: switch carries the new coefficient' if st[a]['max_it'] >= target else 'resume same target'}")
            if st[a]["max_it"] < target or not st[a]["state_match"]:
                if not stop_arm(a):
                    log("ALERT stop failed on KL trigger; exiting")
                    sys.exit(4)
                state["prev_alive"] = {"m": False, "c": False}
                ok = launch_arm(a, resume=True)
                if not ok:
                    state["launch_failures"] += 1
                    if state["launch_failures"] >= 2:
                        log("ALERT launch failed twice after KL step; exiting")
                        sys.exit(5)
                    log("launch failed once after KL step; retry next poll")
                    return
                state["launch_failures"] = 0
                state["prev_alive"] = {x: (x == a) for x in ("m", "c")}
                return
            # boundary + state_match: fall through to the normal switch,
            # which now launches the next arm at the new coefficient
        else:
            with open(KL_HOLD_PATH, "w") as f:
                f.write(f"second KL trigger: {a} it={rec['it']} "
                        f"kl={rec['kl']} at kl_coef 0.15; ruling required\n")
            log(f"KL_HOLD_SET {a} it={rec['it']} kl={rec['kl']} at "
                f"kl_coef 0.15: parking the line for a ruling")
            if not stop_arm(a):
                log("ALERT stop failed while setting KL HOLD; exiting")
                sys.exit(4)
            state["prev_alive"] = {"m": False, "c": False}
            return
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
