#!/usr/bin/env python3
"""K0 monitoring exam snapshot loop for spot-29 (C1 line).

Every 120s: scp the two live files from spot-29 and push full copies to
sync/h29 (synced/h29/bigeval_live/<tag>.jsonl) whenever line count grew.
Exits when both files reach 64 lines (final push + done marker) or after
a hard cap of 90 minutes.
"""
import base64, hashlib, json, os, subprocess, sys, time

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.join(HERE, "exam_snapshots", "h29")
os.makedirs(SNAP_DIR, exist_ok=True)
LOG = os.path.join(SNAP_DIR, "k0mon-snaploop.log")
DONE = os.path.join(SNAP_DIR, "k0mon-snaploop.done")
TAGS = ["k0_a0_b0", "k0_a1_b0"]
TARGET = 64
SSH = ["ssh", "-i", "/home/hatch/.ssh/id_ed25519", "-o", "IdentitiesOnly=yes",
       "-o", "UserKnownHostsFile=/home/hatch/.ssh/known_hosts",
       "-o", "GlobalKnownHostsFile=/dev/null",
       "-o", "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p"]
SCP = ["scp", "-i", "/home/hatch/.ssh/id_ed25519", "-o", "IdentitiesOnly=yes",
       "-o", "UserKnownHostsFile=/home/hatch/.ssh/known_hosts",
       "-o", "GlobalKnownHostsFile=/dev/null",
       "-o", "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p"]
HOST = "root@100.65.8.27"


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
    r = gh_api.put_file("robertyang87", "carbon-league", path, raw,
                        f"snapshot {tag} {count_lines(local)} lines (k0mon snaploop)",
                        sha=sha, branch="sync/h29")
    return r["commit"]["sha"]


def main():
    pushed = {t: -1 for t in TAGS}
    deadline = time.time() + 90 * 60
    while time.time() < deadline:
        all_done = True
        for tag in TAGS:
            local = os.path.join(SNAP_DIR, f"{tag}.jsonl")
            tmp = local + ".part"
            rc = subprocess.run(
                ["timeout", "-k", "30", "120"] + SCP +
                [f"{HOST}:/root/carbon/data/bigeval_live/{tag}.jsonl", tmp],
                capture_output=True, timeout=180)
            if rc.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                os.replace(tmp, local)
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
                f.write(time.strftime("%F %T") + " both tags at 64\n")
            log("both tags at 64, snaploop done")
            return
        time.sleep(120)
    log("deadline reached, snaploop exits")


if __name__ == "__main__":
    main()
