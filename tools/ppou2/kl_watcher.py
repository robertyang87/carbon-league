#!/usr/bin/env python3
"""ppou2 dual-machine KL watcher (post-split layout, 2026-11-15).

Runs on the Muse VM. Every 60 s reads the newest formal train line of
each arm on its own machine (m on gpu-l20-6, c on the spot machine) via
remote_kl_check.py over ssh, and applies the strategy-group KL discipline:
  - first round of an arm at kl_coef 0.15 -> KL15_CALIBRATION log line
    (KL15_CALIBRATION_HIGH if achieved >= 0.10);
  - any round with achieved >= 0.10 -> KL_REPORT_NEEDED log line
    (report-first obligation; surfaced to the strategy group by the agent);
  - any round with achieved >= 0.13 -> KL_TRIGGER: the arm is STOPPED on
    its machine (whole PGID) and kl_hold.flag is written. Both arms are
    already at kl_coef 0.15, so per the 2026-11-15 10:49 ruling a trigger
    now means HOLD and wait for a ruling -- no self-step.
State: kl_watch_state.json next to this script. Log: kl_watcher.log.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(HERE, "kl_watch_state.json")
HOLD_PATH = os.path.join(HERE, "kl_hold.flag")
LOG_PATH = os.path.join(HERE, "kl_watcher.log")

KEY = "/home/hatch/.ssh/id_ed25519"
KH = "/home/hatch/.ssh/known_hosts"
PROXY = "nc -X connect -x hatch-egress-proxy:3130 %h %p"
HOSTS = {"m": "100.79.251.87", "c": "100.92.60.120"}
ARM_DIRS = {"m": "/root/carbon/data/rl3/ppou2_m", "c": "/root/carbon/data/rl3/ppou2_c"}


def log(msg):
    line = "%s %s" % (time.strftime("%F %T"), msg)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def ssh(host, cmd, timeout=50):
    full = ["ssh", "-i", KEY, "-o", "IdentitiesOnly=yes",
            "-o", "UserKnownHostsFile=" + KH,
            "-o", "ProxyCommand=" + PROXY,
            "-o", "ConnectTimeout=15",
            "root@" + host, cmd]
    out = subprocess.run(full, capture_output=True, text=True,
                         timeout=timeout)
    return out.stdout.strip()


def remote_kl(arm):
    out = ssh(HOSTS[arm], "python3 /root/carbon/remote_kl_check.py")
    for ln in out.splitlines():
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if d.get("arm") == arm and d.get("it") is not None:
            return d
    return None


def stop_arm(arm, why):
    host = HOSTS[arm]
    d = ARM_DIRS[arm]
    out = ssh(host, "cat %s.launch.json 2>/dev/null; ls /root/carbon/data/rl3/%s.launch.json 2>/dev/null" % (d, "ppou2_" + arm))
    pid = None
    try:
        pid = json.loads(out.splitlines()[0]).get("pid")
    except Exception:
        pass
    if not pid:
        out = ssh(host, "pgrep -f 'rl2_ppo_train.py.*ppou2_%s' | head -1" % arm)
        pid = out.splitlines()[0] if out else None
    if pid:
        ssh(host, "kill -TERM -%s 2>/dev/null; kill -TERM %s 2>/dev/null; echo stopped" % (pid, pid))
        log("KL_ACTION stopped arm %s pgid/pid %s (%s)" % (arm, pid, why))
    else:
        log("KL_ACTION could not locate arm %s process (%s)" % (arm, why))


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"last_seen_it": {}, "calib_done": {"m": False, "c": True}}


def save_state(st):
    with open(STATE_PATH, "w") as f:
        json.dump(st, f)


def tick(st):
    for arm in ("m", "c"):
        d = remote_kl(arm)
        if not d:
            continue
        it, kl = d["it"], d["kl"]
        if it <= st["last_seen_it"].get(arm, -1):
            continue
        st["last_seen_it"][arm] = it
        if kl is None:
            continue
        log("round arm=%s it=%s kl=%s kl_coef=%s" % (arm, it, kl, d.get("kl_coef")))
        if not st["calib_done"].get(arm, False):
            st["calib_done"][arm] = True
            tag = "KL15_CALIBRATION_HIGH" if kl >= 0.10 else "KL15_CALIBRATION"
            log("%s arm=%s it=%s kl=%s" % (tag, arm, it, kl))
        elif kl >= 0.10 and kl < 0.13:
            log("KL_REPORT_NEEDED arm=%s it=%s kl=%s" % (arm, it, kl))
        if kl >= 0.13:
            log("KL_TRIGGER arm=%s it=%s kl=%s" % (arm, it, kl))
            with open(HOLD_PATH, "w") as f:
                f.write("arm=%s it=%s kl=%s at %s\n" % (arm, it, kl, time.strftime("%F %T")))
            stop_arm(arm, "achieved KL >= 0.13 at kl_coef 0.15 -> HOLD per ruling")
    save_state(st)


def main():
    st = load_state()
    log("watcher start last_seen=%s calib_done=%s" % (st["last_seen_it"], st["calib_done"]))
    while True:
        try:
            if os.path.exists(HOLD_PATH):
                log("HOLD flag present; watcher idle (awaiting ruling)")
                time.sleep(300)
                continue
            tick(st)
        except Exception as e:
            log("loop error: %r" % (e,))
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
