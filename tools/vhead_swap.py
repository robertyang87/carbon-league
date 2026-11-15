#!/usr/bin/env python3
"""Boundary swap to main a4dd8329 (S16): rl2_ppo_train.py 1f1430b4 + rl2_ppo_cnet2.py 62a9adc7,
--value-head utility via merged init (option ii, eng-group ruling 2026-11-13 23:15).
Fires at the first train boundary after arming (base it at arm time)."""
import json, os, signal, subprocess, sys, time, hashlib

RUN = "/root/carbon/data/rl3/snipe_n1"
CODE = "/root/carbon/code"
NEXT = "/root/carbon/code_next"
LOG = "/root/carbon/data/rl3/snipe_n1.log"
STATUS = os.path.join(RUN, "vhead_swap_status.txt")
GO_FILES = {"rl2_ppo_train.py": "1f1430b47faa7f7d22414946a8d5ab53",
            "rl2_ppo_cnet2.py": "62a9adc79069ca6dfa6f344a439a8310"}
MERGED_INIT = "/root/carbon/data/weights/cnet2_m3n1L_plus_utility_it77.npz"
MERGED_MD5 = "f6a72ad6f72018c3e5756c3273e806e2"
DEADLINE = time.time() + 4 * 3600


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


def eval_landed(it):
    try:
        with open(os.path.join(RUN, "eval.jsonl")) as f:
            return any(json.loads(l).get("it") == it for l in f if l.strip())
    except Exception:
        return False


def trainer_pid():
    out = subprocess.run(["pgrep", "-f", "rl2_ppo_train[.]py"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def util_fp_from_state(path):
    import numpy as np, torch
    st = torch.load(path, map_location="cpu")
    h = hashlib.md5()
    h.update(st["net"]["utility.weight"].numpy().tobytes())
    h.update(st["net"]["utility.bias"].numpy().tobytes())
    return h.hexdigest()


def main():
    base = last_it()
    log(f"WATCHER_START vhead swap base_it={base} staged train={GO_FILES['rl2_ppo_train.py'][:8]} cnet2={GO_FILES['rl2_ppo_cnet2.py'][:8]}")
    while time.time() < DEADLINE:
        it = last_it()
        if it is not None and it > base:
            e = (it // 5) * 5
            if e >= 70 and not eval_landed(e) and it < e + 3:
                log(f"EVAL_WAIT it={it} eval{e} not landed")
                time.sleep(60)
                continue
            break
        time.sleep(45)
    else:
        log("ABORT deadline")
        return
    it = last_it()
    for name, want in GO_FILES.items():
        got = md5(os.path.join(NEXT, name))
        if got != want:
            log(f"ABORT staged md5 mismatch {name}: {got}")
            return
    if md5(MERGED_INIT) != MERGED_MD5:
        log("ABORT merged init md5 mismatch")
        return
    pid = trainer_pid()
    if pid is None:
        log("ABORT no trainer")
        return
    argv = open(f"/proc/{pid}/cmdline", "rb").read().decode().split("\0")
    argv = [a for a in argv if a]
    env = {}
    for kv in open(f"/proc/{pid}/environ", "rb").read().decode().split("\0"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v
    log(f"SWAP_START boundary_it={it} pid={pid}")
    fp_backup_holder = {}
    os.kill(pid, signal.SIGTERM)
    for _ in range(60):
        if trainer_pid() is None:
            break
        time.sleep(2)
    if trainer_pid() is not None:
        log("ABORT trainer did not stop")
        return
    log("TRAINER_STOPPED")
    src = os.path.join(RUN, "state_last.pt")
    bak = os.path.join(RUN, f"state_pre_vhead_it{it}.pt")
    subprocess.run(["cp", src, bak], check=True)
    fp_backup_holder["fp"] = util_fp_from_state(bak)
    log(f"STATE_BACKUP {bak} md5 {md5(bak)} util_fp {fp_backup_holder['fp']}")
    for name in GO_FILES:
        subprocess.run(["cp", os.path.join(CODE, name),
                        os.path.join(RUN, name + ".pre_vhead")], check=True)
        subprocess.run(["cp", os.path.join(NEXT, name), os.path.join(CODE, name)], check=True)
        got = md5(os.path.join(CODE, name))
        log(f"FILE_SWAPPED {name} live md5 {got}")
        if got != GO_FILES[name]:
            log(f"ABORT post-swap md5 mismatch {name}")
            return
    rc = subprocess.run(["/root/carbon/.venv/bin/python", "-m", "py_compile",
                         os.path.join(CODE, "rl2_ppo_train.py"),
                         os.path.join(CODE, "rl2_ppo_cnet2.py")]).returncode
    log(f"PY_COMPILE rc={rc}")
    def set_arg(a, flag, val):
        a = list(a)
        if flag in a:
            a[a.index(flag) + 1] = val
        else:
            a += [flag, val]
        return a
    new_argv = set_arg(argv, "--init", MERGED_INIT)
    new_argv = set_arg(new_argv, "--value-head", "utility")
    logf = open(LOG, "ab")
    p = subprocess.Popen(new_argv, cwd=CODE, env=env, stdout=logf,
                         stderr=subprocess.STDOUT, start_new_session=True)
    log(f"RELAUNCHED pid={p.pid} value_head=utility init=merged_plus_utility")
    for wait_s, tag in ((30, "POSTCHECK30"), (120, "POSTCHECK120")):
        time.sleep(wait_s if tag == "POSTCHECK30" else 90)
        alive = trainer_pid() is not None
        log(f"{tag} alive={alive} last_it={last_it()}")
        if not alive:
            log("RELAUNCH_FAILED")
            return
    ep = os.path.join(RUN, "effective_params.json")
    if os.path.exists(ep):
        log("EFFECTIVE_PARAMS " + open(ep).read().replace("\n", " "))
    log("SWAP_DONE")


if __name__ == "__main__":
    main()
