#!/usr/bin/env python3
"""it95 checkpoint exam snapshot loop for spot-29 (C1 line).

User spec: from exam start, snapshot both live per-game files off-machine
every 120 seconds to sync/h29; exit when both tags reach 200 lines.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.join(HERE, "exam_snapshots", "h29")
os.makedirs(SNAP_DIR, exist_ok=True)
LOG = os.path.join(SNAP_DIR, "it95-snaploop.log")
DONE = os.path.join(SNAP_DIR, "it95-snaploop.done")
TAGS = ["it95_m2", "it95base_m2"]
TARGET = 200
HOST = "root@100.65.8.27"
SSH_OPTS = [
    "-i", "/home/hatch/.ssh/id_ed25519",
    "-o", "IdentitiesOnly=yes",
    "-o", "UserKnownHostsFile=/home/hatch/.ssh/known_hosts",
    "-o", "GlobalKnownHostsFile=/dev/null",
    "-o", "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p",
]


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def count_lines(path):
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "rb") as f:
        for ln in f:
            if ln.strip():
                n += 1
    return n


def push(tag, local):
    path = f"synced/h29/bigeval_live/{tag}.jsonl"
    raw = open(local, "rb").read()
    sha = None
    try:
        cur = gh_api.get_file("robertyang87", "carbon-league", path, ref="sync/h29")
        sha = cur["sha"]
    except Exception:
        sha = None
    r = gh_api.put_file(
        "robertyang87", "carbon-league", path, raw,
        f"snapshot {tag} {count_lines(local)} lines (it95 snaploop)",
        sha=sha, branch="sync/h29")
    return r["commit"]["sha"]


def pull(tag):
    local = os.path.join(SNAP_DIR, f"{tag}.jsonl")
    tmp = local + ".part"
    rc = subprocess.run(
        ["timeout", "-k", "30", "120", "scp"] + SSH_OPTS +
        [f"{HOST}:/root/carbon/data/bigeval_live/{tag}.jsonl", tmp],
        capture_output=True, timeout=180)
    if rc.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
        os.replace(tmp, local)
        return True
    return False


def main():
    pushed = {t: -1 for t in TAGS}
    deadline = time.time() + 60 * 60
    log("snaploop start")
    while time.time() < deadline:
        all_done = True
        for tag in TAGS:
            pull(tag)
            local = os.path.join(SNAP_DIR, f"{tag}.jsonl")
            n = count_lines(local)
            if n != pushed[tag] and n > 0:
                try:
                    c = push(tag, local)
                    pushed[tag] = n
                    log(f"push {tag} lines={n} commit={c[:8]}")
                except Exception as e:
                    log(f"push {tag} FAILED lines={n}: {e}")
            if n < TARGET:
                all_done = False
        if all_done:
            with open(DONE, "w") as f:
                f.write(time.strftime("%F %T") + " both tags at 200\n")
            log("both tags at 200, snaploop done")
            return
        time.sleep(120)
    log("deadline reached, snaploop exits")


if __name__ == "__main__":
    main()
