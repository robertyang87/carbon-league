#!/usr/bin/env python3
"""Boundary swap to main 4acbbaea (S18): five files, --utility-source checkpoint,
--resume-vwarm 1 (critic warmup folded in once, per war-room ruling 00:20),
--init back to ORIGINAL cnet2_m3n1L.npz -- (ii) merged init RETIRED at this swap.
Also completes the (ii) overwrite proof BEFORE stopping: utility fingerprints/L2
of it79 state vs it78 boundary backup vs merged init."""
import json, os, signal, subprocess, sys, time, hashlib

RUN = "/root/carbon/data/rl3/snipe_n1"
CODE = "/root/carbon/code"
NEXT = "/root/carbon/code_next"
LOG = "/root/carbon/data/rl3/snipe_n1.log"
STATUS = os.path.join(RUN, "r18_swap_status.txt")
ORIG_INIT = "/root/carbon/data/weights/cnet2_m3n1L.npz"
MERGED_FP = "6e5af436f7b766aa6a55cf55d40399ea"
GO = {"rl2_ppo_train.py": "c27b7ea5b74e5f0960bae620706fa272",
      "stage0_teacher.py": "c6ade5ced42e2c54a77c1499163aa151",
      "stage1_teacher_gate.py": "4969a9fc0c374e17cfe54eadc24dada9",
      "cnet2_build.py": "3b99ae47699eff7e232a83790956f757",
      "producer/selfplay_gen.py": "523963c2a7e6d14880c948b8b028a158"}
OLD = {"rl2_ppo_train.py": "1f1430b47faa7f7d22414946a8d5ab53",
       "stage0_teacher.py": "77ba36773fb135c92a6e5f171f88bdbd",
       "stage1_teacher_gate.py": "13d89867a17d153a41ea9e7914e8ac90",
       "cnet2_build.py": "030b317bc649fd45c07249c5e3bb662f",
       "producer/selfplay_gen.py": "95d2e29fbec8cfcebee57bf9bf0d25ed",
       "rl2_ppo_cnet2.py": "62a9adc79069ca6dfa6f344a439a8310"}
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


def util_tensors(path):
    import torch
    st = torch.load(path, map_location="cpu")
    return (st["net"]["utility.weight"].numpy().astype("float64"),
            st["net"]["utility.bias"].numpy().astype("float64"))


def util_fp(t):
    h = hashlib.md5()
    h.update(t[0].tobytes()); h.update(t[1].tobytes())
    return h.hexdigest()


def main():
    base = last_it()
    log(f"WATCHER_START r18 swap base_it={base} staged train={GO['rl2_ppo_train.py'][:8]}")
    while time.time() < DEADLINE:
        it = last_it()
        if it is not None and it > base:
            e = (it // 5) * 5
            if e >= 70 and not eval_landed(e) and it < e + 3:
                log(f"EVAL_WAIT it={it} eval{e} not landed")
                time.sleep(60)
                continue
            break
        time.sleep(30)
    else:
        log("ABORT deadline")
        return
    it = last_it()
    # --- (ii) overwrite proof, pre-stop: it79 state vs it78 backup vs merged init
    try:
        import numpy as np
        t79 = util_tensors(os.path.join(RUN, "state_last.pt"))
        t78 = util_tensors(os.path.join(RUN, "state_pre_vhead_it78.pt"))
        d_79_78 = float(np.linalg.norm(t79[0] - t78[0]) + np.linalg.norm(t79[1] - t78[1]))
        log(f"OVERWRITE_PROOF it{it} util_fp={util_fp(t79)} it78bak_fp={util_fp(t78)} merged_fp={MERGED_FP}")
        log(f"OVERWRITE_PROOF L2(it{it},it78bak)={d_79_78:.6g} (one round of head updates; merged init is one round BEHIND the it78 backup, so a resume from merged values would sit ~2 rounds away)")
    except Exception as ex:
        log(f"OVERWRITE_PROOF failed: {ex}")
    for rel, want in GO.items():
        got = md5(os.path.join(NEXT, rel))
        if got != want:
            log(f"ABORT staged md5 mismatch {rel}: {got}")
            return
    for rel, want in OLD.items():
        got = md5(os.path.join(CODE, rel))
        if got != want:
            log(f"ABORT live drift {rel}: {got} want {want}")
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
    os.kill(pid, signal.SIGTERM)
    for _ in range(60):
        if trainer_pid() is None:
            break
        time.sleep(2)
    if trainer_pid() is not None:
        log("ABORT trainer did not stop")
        return
    log("TRAINER_STOPPED")
    bak = os.path.join(RUN, f"state_pre_r18_it{it}.pt")
    subprocess.run(["cp", os.path.join(RUN, "state_last.pt"), bak], check=True)
    log(f"STATE_BACKUP {bak} md5 {md5(bak)}")
    for rel in GO:
        subprocess.run(["cp", os.path.join(CODE, rel),
                        os.path.join(RUN, os.path.basename(rel) + ".pre_r18")], check=True)
        subprocess.run(["cp", os.path.join(NEXT, rel), os.path.join(CODE, rel)], check=True)
        got = md5(os.path.join(CODE, rel))
        log(f"FILE_SWAPPED {rel} live md5 {got}")
        if got != GO[rel]:
            log(f"ABORT post-swap md5 mismatch {rel}")
            return
    rc = subprocess.run(["/root/carbon/.venv/bin/python", "-m", "py_compile"] +
                        [os.path.join(CODE, r) for r in GO]).returncode
    log(f"PY_COMPILE rc={rc}")

    def set_arg(a, flag, val):
        a = list(a)
        if flag in a:
            a[a.index(flag) + 1] = val
        else:
            a += [flag, val]
        return a
    new_argv = set_arg(argv, "--init", ORIG_INIT)
    new_argv = set_arg(new_argv, "--utility-source", "checkpoint")
    new_argv = set_arg(new_argv, "--resume-vwarm", "1")
    logf = open(LOG, "ab")
    p = subprocess.Popen(new_argv, cwd=CODE, env=env, stdout=logf,
                         stderr=subprocess.STDOUT, start_new_session=True)
    log(f"RELAUNCHED pid={p.pid} utility_source=checkpoint resume_vwarm=1 init=ORIGINAL (ii retired)")
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
