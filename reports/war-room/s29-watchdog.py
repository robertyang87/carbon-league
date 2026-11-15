"""Watchdog for the four h29 backfill loops (C1, 2026-11-13).

Why: a VM/deployment replacement at ~02:00 CST killed all four nohup
loops at once (uptime showed the VM was 16 min old; no exit lines,
home dir intact). nohup does not survive a VM replacement.
This script re-launches any missing loop; a cron runs it every 10 min.

Rules:
- Only relaunch while the backfill is unfinished (any tag < 200 seeds)
  and before the war-room DEADLINE (1794546523 = 13:08:43 CST).
  After completion or deadline, do nothing (never resurrect a finished
  loop, and never extend the deadline without a new ruling).
- Match processes by their exact absolute command line; the watchdog's
  own cmdline never contains a loop name, so no self-match.
- Read-only with respect to the loops' stores; relaunch only.
- Freeze flag (2026-11-13 ~06:19, third VM replacement): when
  h29_freeze.flag exists in BASE, a relaunched loop is SIGSTOPped
  immediately and any running loop not already stopped is stopped too,
  so a replacement never resurrects the rings in running state while
  the channel outage freeze is in force. Delete the flag file to end
  the freeze (war room does this when it broadcasts channel recovery,
  C1 SIGCONTs the rings after).
"""
import json, os, signal, subprocess, sys, time

BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
SNAP = BASE + "/exam_snapshots/h29"
DEADLINE = 1794546523  # 13:08:43 CST, war-room ruling 01:26
LOOPS = {
    "s29-snaploop-fable.py": "snaploop-fable.log",
    "s29-snaploop-fableg1.py": "snaploop-fableg1.log",
    "s29-snaploop-m7v2r1.py": "snaploop-m7v2r1.log",
    "s29-snaploop-m7v2g1.py": "snaploop-m7v2g1.log",
}
STORES = ["lines-fable.json", "lines-fableg1.json", "lines-m7v2r1.json", "lines-m7v2g1.json"]
WLOG = SNAP + "/snaploop-watchdog.log"
FREEZE = BASE + "/h29_freeze.flag"


def wlog(s):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
    print(line, flush=True)
    with open(WLOG, "a") as f:
        f.write(line + "\n")


def complete():
    try:
        for name in STORES:
            d = json.load(open(os.path.join(SNAP, name)))
            for tag, rows in d.items():
                if len(rows) < 200:
                    return False
        return True
    except Exception:
        return False


def running(script):
    # Match both launch styles: relative ("python3 s29-snaploop-x.py")
    # and absolute ("python3 /home/hatch/.../s29-snaploop-x.py").
    # pgrep excludes itself; the watchdog's own cmdline never contains
    # a loop name, so no self-match. Anchor both ends so that e.g.
    # s29-snaploop-fable.py never matches s29-snaploop-fableg1.py.
    base = script.replace(".", r"\.")
    pat = f"^python3 (.*/)?{base}$"
    r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    return bool(r.stdout.strip())


def enforce_freeze():
    """While the freeze flag exists, every ring must be in T state.

    Called after any relaunch, so a VM replacement can never leave a
    ring running during the freeze. Already-stopped rings are left
    alone; a ring that vanished is relaunched first (in main) and then
    stopped here.
    """
    if not os.path.exists(FREEZE):
        return
    for script in LOOPS:
        base = script.replace(".", r"\.")
        pat = f"^python3 (.*/)?{base}$"
        r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
        for pid_s in r.stdout.split():
            try:
                pid = int(pid_s)
                with open(f"/proc/{pid}/stat") as f:
                    state = f.read().rsplit(") ", 1)[1].split()[0]
                if state != "T":
                    os.kill(pid, signal.SIGSTOP)
                    wlog(f"freeze: SIGSTOP {script} pid={pid} (state was {state})")
            except Exception as e:
                wlog(f"freeze: could not stop {script} pid={pid_s}: {e}")


def main():
    if complete():
        wlog("all tags complete, watchdog idle")
        return
    if time.time() >= DEADLINE:
        wlog("past DEADLINE, watchdog idle (no extension without a ruling)")
        return
    for script, logf in LOOPS.items():
        if running(script):
            continue
        wlog(f"RELAUNCH missing loop {script}")
        with open(os.path.join(SNAP, logf), "ab") as lf:
            subprocess.Popen(
                ["python3", f"{BASE}/{script}"],
                cwd=BASE, stdout=subprocess.DEVNULL, stderr=lf,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        time.sleep(1)
        wlog(f"relaunched {script}, running={running(script)}")
    enforce_freeze()


if __name__ == "__main__":
    main()
