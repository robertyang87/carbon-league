#!/usr/bin/env python3
"""S19 mitigation: rolling backup of state_last.pt, starting it81.
After each train round lands (LASTIT >= 81), copy state_last.pt to
state_last.pt.prev1 (shifting the old prev1 to .prev2), verify size+md5
match the source, log to state_backup_status.txt. No training change.
Runs until killed; first backup receipt is reported by the operator."""
import hashlib, json, os, shutil, subprocess, time

RUN = "/root/carbon/data/rl3/snipe_n1"
SRC = os.path.join(RUN, "state_last.pt")
PREV1 = SRC + ".prev1"
PREV2 = SRC + ".prev2"
STATUS = os.path.join(RUN, "state_backup_status.txt")
DEADLINE = time.time() + 72 * 3600


def log(msg):
    with open(STATUS, "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def last_it():
    try:
        last = None
        with open(os.path.join(RUN, "train.jsonl")) as f:
            for line in f:
                if line.strip():
                    last = line
        return json.loads(last)["it"] if last else -1
    except Exception:
        return -1


def file_stable(path, wait=30):
    """Two size+mtime reads `wait` apart; True if unchanged."""
    try:
        s1 = os.stat(path)
    except OSError:
        return False
    time.sleep(wait)
    try:
        s2 = os.stat(path)
    except OSError:
        return False
    return (s1.st_size, s1.st_mtime_ns) == (s2.st_size, s2.st_mtime_ns)


def backup(it):
    for attempt in range(5):
        if not file_stable(SRC, 30):
            log(f"it{it} attempt{attempt}: state still changing, retry")
            continue
        size = os.path.getsize(SRC)
        src_md5 = md5(SRC)
        if os.path.exists(PREV1):
            shutil.move(PREV1, PREV2)
        shutil.copyfile(SRC, PREV1)
        got_size = os.path.getsize(PREV1)
        got_md5 = md5(PREV1)
        if got_size == size and got_md5 == src_md5:
            log(f"BACKUP_OK it{it} {PREV1} size={size} md5={got_md5} (prev2 rotated)")
            return True
        log(f"BACKUP_MISMATCH it{it} attempt{attempt} src={src_md5} copy={got_md5}")
        time.sleep(60)
    log(f"BACKUP_FAILED it{it} after 5 attempts")
    return False


def main():
    done = -1
    log("WATCHER_START s19 rolling state backup (from it81)")
    while time.time() < DEADLINE:
        it = last_it()
        if it is not None and it >= 81 and it > done:
            backup(it)
            done = it
        time.sleep(60)


if __name__ == "__main__":
    main()
