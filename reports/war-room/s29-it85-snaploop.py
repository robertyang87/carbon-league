#!/usr/bin/env python3
"""s29 it85 checkpoint-1 exam snaploop: mirror spot-29 bigeval_live files to
sync/h29 every 120s (exact machine bytes), until both main fields reach 200
lines and are pushed. Written fresh per C1 design口径 (2026-11-14)."""
import os, sys, time, hashlib, subprocess, datetime

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

O, R = "robertyang87", "carbon-league"
BRANCH = "sync/h29"
BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h29"
os.makedirs(BASE, exist_ok=True)
LOG = os.path.join(BASE, "it85-snaploop.log")
DONE = os.path.join(BASE, "it85-snaploop.done")
REMOTE = "root@100.65.8.27:/root/carbon/data/bigeval_live/"
SCP = ["scp", "-i", "/home/hatch/.ssh/id_ed25519", "-o", "IdentitiesOnly=yes",
       "-o", "UserKnownHostsFile=/home/hatch/.ssh/known_hosts",
       "-o", "GlobalKnownHostsFile=/dev/null",
       "-o", "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p"]
FILES = {  # local mirror name -> (repo path, target lines or None)
    "it85_smoke.jsonl": ("synced/h29/bigeval_live/it85_smoke.jsonl", None),
    "it85base_smoke.jsonl": ("synced/h29/bigeval_live/it85base_smoke.jsonl", None),
    "it85_m2.jsonl": ("synced/h29/bigeval_live/it85_m2.jsonl", 200),
    "it85base_m2.jsonl": ("synced/h29/bigeval_live/it85base_m2.jsonl", 200),
}
DEADLINE = time.time() + 4 * 3600


def now():
    return datetime.datetime.now().strftime("%F %T CST")


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{now()} {msg}\n")


def lines_of(path):
    try:
        with open(path, "rb") as f:
            return f.read().count(b"\n")
    except FileNotFoundError:
        return -1


def push(local, repo_path):
    data = open(local, "rb").read()
    sha = None
    try:
        cur = gh_api.get_file(O, R, repo_path, ref=BRANCH)
        sha = cur["sha"]
    except Exception:
        sha = None
    gh_api.put_file(O, R, repo_path, data,
                    f"sync h29 it85 snap {os.path.basename(local)}",
                    sha=sha, branch=BRANCH)
    return hashlib.md5(data).hexdigest(), data.count(b"\n")


def main():
    log(f"SNAPLOOP_START pid={os.getpid()}")
    pushed = {}  # name -> md5 last pushed
    while time.time() < DEADLINE:
        all_done = True
        for name, (repo_path, target) in FILES.items():
            local = os.path.join(BASE, name)
            try:
                subprocess.run(SCP + [REMOTE + name, local],
                               timeout=100, capture_output=True)
            except Exception as e:
                log(f"SCP_FAIL {name}: {e}")
                continue
            n = lines_of(local)
            if n < 0:
                if target:
                    all_done = False
                continue
            data = open(local, "rb").read()
            md5 = hashlib.md5(data).hexdigest()
            if pushed.get(name) != md5:
                try:
                    pmd5, pn = push(local, repo_path)
                    pushed[name] = pmd5
                    log(f"PUSH {name} lines={pn} md5={pmd5[:8]}")
                except Exception as e:
                    log(f"PUSH_FAIL {name}: {e}")
                    if target:
                        all_done = False
                    continue
            if target and n < target:
                all_done = False
        if all_done:
            log("ALL_TARGETS_200_PUSHED snaploop exit")
            with open(DONE, "w") as f:
                f.write(f"{now()} complete\n")
            return
        time.sleep(120)
    log("DEADLINE_EXIT snaploop")


if __name__ == "__main__":
    main()
