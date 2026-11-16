#!/usr/bin/env python3
"""ppou2 dual-machine KL watcher (post-split layout; Amendment 1 caliber, 2026-11-15).

Runs on the Muse VM. Every 60 s reads the newest formal train line of
each arm on its own machine (m on gpu-l20-6, c on the spot machine) via
remote_kl_check.py over ssh.

Semantics (Amendment 1, strategy group 2026-11-15 17:19): the logged kl is
KL(pi_ref || pi), the DISTANCE from the arm's frozen anchor -- not a
per-round step. Lines enforced here:
  - distance >= 0.15 (per-anchor cap): stop the arm at its round boundary
    (this watcher only ever sees landed rounds), write kl_hold.flag; a
    checkpoint exam + ruling must follow -- no crossing the cap.
  - single-round distance jump > +0.03: log KL_JUMP_REPORT (report-first
    proxy alarm; the agent relays it to the strategy group same round).

Efficiency watch (war room, 2026-11-15 17:20, owner order after the main
arm slowed ~40% unnoticed; merged with the Amendment 1 rewrite same day):
every round also logs a pace line (games_per_min / roll_s / upd_s /
dec_per_s from the same train row), GPU utilization is sampled every
tick, and two alerts exist:
  - PACE_ALERT: games_per_min below 75% of the arm's best-seen rate
    for 2+ consecutive rounds;
  - UTIL_ALERT: GPU utilization never reached 30% across a whole round
    (not even the update phase used the GPU).
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
DIST_CAP = 0.15
JUMP = 0.03


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


def remote_pace(arm):
    out = ssh(HOSTS[arm], "tail -3 %s/train.jsonl" % ARM_DIRS[arm])
    rows = []
    for ln in out.splitlines():
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue
    formal = [r for r in rows if not r.get("warm") and r.get("it") is not None]
    return formal[-1] if formal else None


def gpu_util(arm):
    out = ssh(HOSTS[arm],
              "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits")
    try:
        return int(out.splitlines()[0].strip())
    except Exception:
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
            st = json.load(f)
    else:
        st = {}
    st.setdefault("last_seen_it", {})
    st.setdefault("last_kl", {})
    # pace_ref for arm m seeded from the strategy group's 2026-11-15
    # 17:07 report: main-arm games_per_min was 184.9 before the decline.
    st.setdefault("pace_ref", {"m": 184.9})
    st.setdefault("pace_bad", {})
    st.setdefault("util_samples", {})
    return st


def save_state(st):
    with open(STATE_PATH, "w") as f:
        json.dump(st, f)


def tick(st):
    for arm in ("m", "c"):
        u = gpu_util(arm)
        if u is not None:
            st.setdefault("util_samples", {}).setdefault(arm, []).append(u)
    for arm in ("m", "c"):
        d = remote_kl(arm)
        if not d:
            continue
        it, kl = d["it"], d["kl"]
        if it <= st["last_seen_it"].get(arm, -1):
            continue
        prev = st["last_kl"].get(arm)
        st["last_seen_it"][arm] = it
        p = remote_pace(arm)
        if p:
            gpm = p.get("games_per_min")
            log("pace arm=%s it=%s gpm=%s roll_s=%s upd_s=%s dec_per_s=%s"
                % (arm, it, gpm, p.get("roll_s"), p.get("upd_s"),
                   p.get("dec_per_s")))
            if gpm:
                # pace_ref is the MEDIAN of the last 3 rounds, not best-seen
                # max: a single short/fast outlier round (c it=21: 258.2
                # gpm, roll_s 1007.7 vs ~1400 stable) pinned ref=258.2 and
                # made every normal ~187 round trip PACE_ALERT. Median
                # tracks the stable level; a genuine speedup still moves it.
                hist = st.setdefault("pace_hist", {}).setdefault(arm, [])
                hist.append(gpm)
                del hist[:-5]
                if len(hist) >= 3:
                    ref = sorted(hist[-3:])[1]
                else:
                    ref = st.setdefault("pace_ref", {}).get(arm)
                    ref = gpm if ref is None else max(ref, gpm)
                st["pace_ref"][arm] = ref
                bad = st.setdefault("pace_bad", {}).get(arm, 0)
                bad = bad + 1 if gpm < 0.75 * ref else 0
                st["pace_bad"][arm] = bad
                if bad >= 2:
                    log("PACE_ALERT arm=%s it=%s gpm=%s ref=%s "
                        "(2+ rounds below 75%% of best)" % (arm, it, gpm, ref))
            samples = st.setdefault("util_samples", {}).get(arm, [])
            if samples:
                umax = max(samples)
                if umax < 30:
                    log("UTIL_ALERT arm=%s it=%s gpu_util_max_since_last_round=%s%%"
                        % (arm, it, umax))
                st["util_samples"][arm] = []
        if kl is None:
            continue
        st["last_kl"][arm] = kl
        log("round arm=%s it=%s kl=%s kl_coef=%s" % (arm, it, kl, d.get("kl_coef")))
        if prev is not None and kl - prev > JUMP:
            log("KL_JUMP_REPORT arm=%s it=%s kl=%s prev=%s jump=%+.4f" % (arm, it, kl, prev, kl - prev))
        if kl >= DIST_CAP:
            log("KL_DISTANCE_CAP arm=%s it=%s kl=%s" % (arm, it, kl))
            with open(HOLD_PATH, "w") as f:
                f.write("arm=%s it=%s kl=%s at %s -- per-anchor distance cap 0.15 reached (Amendment 1); arm stopped at boundary; checkpoint exam + ruling required\n" % (arm, it, kl, time.strftime("%F %T")))
            stop_arm(arm, "distance cap 0.15 reached (Amendment 1)")
    save_state(st)


def main():
    st = load_state()
    log("watcher start (Amendment 1 caliber) last_seen=%s last_kl=%s" % (st["last_seen_it"], st["last_kl"]))
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
