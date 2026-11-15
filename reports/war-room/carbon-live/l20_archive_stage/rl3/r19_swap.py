#!/usr/bin/env python3
"""§19 boundary swap watcher for snipe_n1 (L20 line, sole executor).

Waits for the it81 train row, backs up state, kills the trainer's whole
process group (orphan-worker protocol), verifies the three acceptance
checks, swaps the code set to main 2523235, relaunches with the exact
argv/env captured from /proc, then waits for the it82 row plus the new
code's first atomic save (state_previous.pt + checkpoint_complete.json)
before retiring the §19 rolling-backup watcher (PID 33517).
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

RUN = "/root/carbon/data/rl3/snipe_n1"
CODE = "/root/carbon/code"
STAGE = "/root/carbon/code_next"
TRAIN = os.path.join(RUN, "train.jsonl")
STATE = os.path.join(RUN, "state_last.pt")
BACKUP_STATE = os.path.join(RUN, "state_pre_r19_it82.pt")
STATUS = os.path.join(RUN, "r19_swap_status.txt")
LOG = "/root/carbon/data/rl3/snipe_n1.log"
BACKUP_WATCHER_PID = 33517
FILES = {
    "rl2_ppo_train.py": "cf572ec15903db866e3df167c6e88df0",
    "stage1_teacher_gate.py": "aae25fdad5c376b0d4b907ace0f1e31f",
    "cnet2_build.py": "46c8a5380e1c354beb2db68ba144da9e",
    "rl2_checkpoint.py": "fcf2edb1c0eb9a649bcccaff8ac3921b",
}
UNCHANGED = {
    "stage0_teacher.py": "c6ade5ced42e2c54a77c1499163aa151",
    os.path.join("producer", "selfplay_gen.py"): "523963c2a7e6d14880c948b8b028a158",
}


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    with open(STATUS, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def last_it():
    try:
        with open(TRAIN, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200000))
            tail = f.read().decode("utf-8", "replace")
        for line in reversed(tail.strip().splitlines()):
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict) and "it" in row:
                return row
    except FileNotFoundError:
        pass
    return None


def find_trainer():
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                parts = f.read().split(b"\0")
            args = [p.decode("utf-8", "replace") for p in parts if p]
        except Exception:
            continue
        if args and "rl2_ppo_train.py" in " ".join(args) and "snipe_n1" in " ".join(args):
            return int(pid), args
    return None, None


def proc_env(pid):
    env = {}
    with open(f"/proc/{pid}/environ", "rb") as f:
        for item in f.read().split(b"\0"):
            if b"=" in item:
                k, v = item.split(b"=", 1)
                env[k.decode()] = v.decode()
    return env


def group_members(pgid):
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                stat = f.read().decode("utf-8", "replace")
            pgrp = int(stat.rsplit(")", 1)[1].split()[2])
            if pgrp == pgid:
                out.append(int(pid))
        except Exception:
            continue
    return out


def spawn_main_pids():
    out = []
    me = os.getpid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().decode("utf-8", "replace")
            if "spawn_main" in cmd:
                out.append(int(pid))
        except Exception:
            continue
    return out


def gpu_apps():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    except Exception as e:
        return [f"ERR {e}"]


def gpu_used_mib():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def state_stable_it():
    """Return embedded it once state_last.pt is stable across two reads."""
    prev = None
    for _ in range(40):
        try:
            st = os.stat(STATE)
            sig = (st.st_size, st.st_mtime)
        except FileNotFoundError:
            sig = None
        if sig is not None and sig == prev:
            import torch
            st8 = torch.load(STATE, map_location="cpu", weights_only=False)
            return int(st8["it"])
        prev = sig
        time.sleep(30)
    return None


def main():
    log("WATCHER_START r19 boundary swap armed (base_it=81)")
    pid, argv = find_trainer()
    if pid is None:
        log("ABORT no trainer found at arm time")
        return
    pgid = os.getpgid(pid)
    env = proc_env(pid)
    log(f"trainer pid={pid} pgid={pgid} argv0={argv[0]} nargs={len(argv)}")

    # 1) wait for it81 row
    beat = time.time()
    while True:
        row = last_it()
        if row and row.get("it", 0) >= 81:
            log(f"it81 row seen kl={row.get('kl')} kl_coef={row.get('kl_coef')} wr={row.get('winrate')}")
            break
        if time.time() - beat > 300:
            log(f"waiting it81 (last_it={row.get('it') if row else None})")
            beat = time.time()
        time.sleep(20)

    # 2) state saved + stable, embedded it must be 81
    it_embedded = state_stable_it()
    log(f"state stable, embedded it={it_embedded}")
    if it_embedded != 81:
        log(f"ABORT embedded it {it_embedded} != 81")
        return

    # 3) state backup with md5 verification
    subprocess.run(["cp", "-f", STATE, BACKUP_STATE], check=True)
    src_md5, bak_md5 = md5(STATE), md5(BACKUP_STATE)
    log(f"state backup {BACKUP_STATE} md5={bak_md5} src_md5={src_md5} match={src_md5 == bak_md5}")
    if src_md5 != bak_md5:
        log("ABORT state backup md5 mismatch")
        return

    # 4) kill the whole process group
    members_before = group_members(pgid)
    log(f"group members before kill: {len(members_before)} pids")
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.time() + 120
    while time.time() < deadline and group_members(pgid):
        time.sleep(5)
    left = group_members(pgid)
    if left:
        log(f"SIGTERM left {left}, sending SIGKILL")
        os.killpg(pgid, signal.SIGKILL)
        deadline = time.time() + 60
        while time.time() < deadline and group_members(pgid):
            time.sleep(5)

    # 5) three acceptance checks
    left = group_members(pgid)
    spawns = spawn_main_pids()
    apps = gpu_apps()
    used = gpu_used_mib()
    log(f"post-kill checks: group_left={left} spawn_main={spawns} gpu_apps={apps} gpu_used_mib={used}")
    if left or spawns or apps:
        log("ABORT_DIRTY acceptance checks failed; NOT relaunching")
        return
    log("acceptance checks PASS (group empty, spawn_main=0, gpu apps empty)")

    # 6) swap code files (backups as *.pre_r19), verify md5 + py_compile
    for name, want in FILES.items():
        live = os.path.join(CODE, name)
        staged = os.path.join(STAGE, name)
        if md5(staged) != want:
            log(f"ABORT staged {name} md5 mismatch")
            return
        if os.path.exists(live):
            subprocess.run(["cp", "-f", live, live + ".pre_r19"], check=True)
        subprocess.run(["cp", "-f", staged, live], check=True)
        got = md5(live)
        log(f"swapped {name} md5={got} ok={got == want}")
        if got != want:
            log(f"ABORT live {name} md5 mismatch after copy")
            return
    for name, want in UNCHANGED.items():
        got = md5(os.path.join(CODE, name))
        log(f"unchanged {name} md5={got} ok={got == want}")
    r = subprocess.run(
        ["/root/carbon/.venv/bin/python", "-m", "py_compile"] +
        [os.path.join(CODE, n) for n in FILES],
        capture_output=True, text=True)
    log(f"py_compile rc={r.returncode} {r.stderr.strip()[:200]}")
    if r.returncode != 0:
        log("ABORT py_compile failed; NOT relaunching")
        return

    # 7) relaunch with captured argv/env, own process group
    logf = open(LOG, "ab")
    p = subprocess.Popen(argv, cwd=CODE, env=env, stdout=logf,
                         stderr=subprocess.STDOUT, start_new_session=True)
    log(f"RELAUNCHED pid={p.pid} pgid={os.getpgid(p.pid)}")
    for delay in (30, 90):
        time.sleep(delay)
        alive = True
        try:
            os.kill(p.pid, 0)
        except ProcessLookupError:
            alive = False
        log(f"postcheck +{delay}s alive={alive} gpu_used_mib={gpu_used_mib()}")
        if not alive:
            log("RELAUNCH_FAILED new trainer died; NOT retrying")
            return
    try:
        with open(os.path.join(RUN, "effective_params.json")) as f:
            ep = json.load(f)
        keep = {k: ep.get(k) for k in ("it0", "reward", "gamma", "rscale", "kl_coef",
                                        "value_head", "critic_warmup_iters", "u_mode_problems")}
        keep["utility_provenance"] = ep.get("utility_provenance")
        log("effective_params " + json.dumps(keep, ensure_ascii=False))
    except Exception as e:
        log(f"effective_params read failed: {e}")

    # 8) wait it82 row + first atomic save, then retire rolling-backup watcher
    beat = time.time()
    while True:
        row = last_it()
        prev_ok = os.path.exists(os.path.join(RUN, "state_previous.pt"))
        cc_ok = os.path.exists(os.path.join(RUN, "checkpoint_complete.json"))
        if row and row.get("it", 0) >= 82 and prev_ok and cc_ok:
            log(f"it82 row kl={row.get('kl')} kl_coef={row.get('kl_coef')} wr={row.get('winrate')} warm={row.get('warm')}")
            log("atomic save verified: state_previous.pt + checkpoint_complete.json present")
            break
        if time.time() - beat > 300:
            log(f"waiting it82/atomic (last_it={row.get('it') if row else None} prev={prev_ok} cc={cc_ok})")
            beat = time.time()
        time.sleep(60)
    try:
        os.kill(BACKUP_WATCHER_PID, signal.SIGTERM)
        time.sleep(3)
        try:
            os.kill(BACKUP_WATCHER_PID, 0)
            log(f"backup watcher {BACKUP_WATCHER_PID} still alive after SIGTERM")
        except ProcessLookupError:
            log(f"backup watcher {BACKUP_WATCHER_PID} retired at " +
                time.strftime("%Y-%m-%d %H:%M:%S"))
    except ProcessLookupError:
        log(f"backup watcher {BACKUP_WATCHER_PID} already gone")
    log("SWAP_COMPLETE")


if __name__ == "__main__":
    main()
