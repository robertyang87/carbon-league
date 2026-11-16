"""Studio ppou2 probe watcher.

Stop rule (filed): first train.jsonl row with warm==false (a real round,
not a resume-warmup row) AND out/state_last.pt mtime newer than launch
start -> SIGTERM the whole process group, verify empty, record status.

Also tracks peak total RSS of the group (poll every 20 s) and writes
out/probe_watch.json on every poll so progress is externally readable.
Hard cap: 120 min without a full round -> SIGTERM, then SIGKILL, status
TIMEOUT_KILLED_120MIN (a round that slow already fails the 45-min bar).
Runs on the Studio itself; started with nohup so it survives ssh exit.
"""
import json
import os
import signal
import subprocess
import sys
import time

P = "/Users/arm/junyang/carbon/data/ppou2_probe_studio"
OUT = os.path.join(P, "out")
STATUS = os.path.join(OUT, "probe_watch.json")
TRAIN_JSONL = os.path.join(OUT, "train.jsonl")
STATE_LAST = os.path.join(OUT, "state_last.pt")

pgid = int(sys.argv[1])
start = float(sys.argv[2])
peak = 0.0


def group_rss_mb():
    """(total RSS MB, process count) for pgid via ps."""
    try:
        r = subprocess.run(["ps", "-ax", "-o", "pgid=", "-o", "rss="],
                           capture_output=True, text=True, timeout=15)
        total = 0
        n = 0
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                g, rss = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if g == pgid:
                total += rss
                n += 1
        return total / 1024.0, n
    except Exception:
        return 0.0, 0


def group_alive():
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def first_real_row():
    if not os.path.exists(TRAIN_JSONL):
        return None
    with open(TRAIN_JSONL, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("warm") is False:
                return rec
    return None


def train_rows():
    if not os.path.exists(TRAIN_JSONL):
        return 0
    with open(TRAIN_JSONL, errors="replace") as f:
        return sum(1 for _ in f)


def write_status(**kw):
    st = {"t": time.strftime("%F %T %Z"), "pgid": pgid,
          "peak_rss_mb": round(peak, 1), **kw}
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATUS)
    return st


def stop_group():
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        return
    time.sleep(10)
    if group_alive():
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        time.sleep(5)


t0 = time.time()
while True:
    rss, nproc = group_rss_mb()
    peak = max(peak, rss)
    alive = group_alive()
    row = first_real_row()
    state_ok = os.path.exists(STATE_LAST) and os.path.getmtime(STATE_LAST) > start
    elapsed = time.time() - t0
    if row is not None and state_ok:
        stop_group()
        write_status(status="STOPPED_ROUND_DONE",
                     elapsed_s=round(time.time() - t0, 1),
                     first_real_row=row,
                     state_mtime=time.strftime(
                         "%F %T %Z",
                         time.localtime(os.path.getmtime(STATE_LAST))),
                     group_empty=not group_alive(),
                     rss_mb_at_stop=round(rss, 1), nproc=nproc)
        break
    if not alive:
        write_status(status="GROUP_GONE",
                     elapsed_s=round(elapsed, 1),
                     first_real_row=row, state_ok=state_ok, nproc=nproc)
        break
    if elapsed > 120 * 60:
        stop_group()
        write_status(status="TIMEOUT_KILLED_120MIN",
                     elapsed_s=round(time.time() - t0, 1),
                     first_real_row=row, state_ok=state_ok,
                     group_empty=not group_alive())
        break
    write_status(status="RUNNING", elapsed_s=round(elapsed, 1),
                 rss_mb=round(rss, 1), nproc=nproc,
                 train_rows=train_rows(),
                 first_real_row=row, state_ok=state_ok)
    time.sleep(20)
print("WATCHER_DONE")
