#!/usr/bin/env python3
"""§15 boundary execution watcher: KL 0.15 -> 0.12 + pool 50/25/25 re-tier.

Armed after the it85 read (block kl 0.0787-0.0894, never >= 0.12; EVAL not
collapsed). Executes at the it86 -> it87 boundary: state backup, whole
process-group kill (orphan protocol), three acceptance checks, relaunch
with captured argv modified in exactly two places (--kl 0.12, --pool
re-tiered weights, total weight unchanged at 5.5), then verifies the
it87 row carries kl_coef 0.12.
"""
import hashlib
import json
import os
import signal
import subprocess
import time

RUN = "/root/carbon/data/rl3/snipe_n1"
CODE = "/root/carbon/code"
TRAIN = os.path.join(RUN, "train.jsonl")
STATE = os.path.join(RUN, "state_last.pt")
BACKUP_STATE = os.path.join(RUN, "state_pre_kl012_it87.pt")
STATUS = os.path.join(RUN, "kl012_swap_status.txt")
LOG = "/root/carbon/data/rl3/snipe_n1.log"
BASE_IT = 86

# argv order preserved from the live pool; weights re-tiered 50/25/25.
POOL = [
    ("SNAP:cnet2:/root/carbon/data/weights/cnet2_m3s2g1L.npz#T2", 0.55),          # T1
    ("SNAP:cnet2:/root/carbon/data/weights/cnet2_m6r1L.npz#T2", 0.45),            # T3
    ("SNAP:cnet2:/root/carbon/data/weights/cnet2_m3s2r1L.npz#T2", 1.10),          # T1
    ("SNAP:cnet2:/root/carbon/data/weights/cnet2_m7bsr1L.npz#T2", 0.55),          # T1
    ("SNAP:cnet2:/root/carbon/data/weights/cnet2_m9xj260wr1_fromstudio.npz#T2", 0.50),  # T2
    ("SNAP:cnet2:/root/carbon/data/weights/cnet2_m9xj260r1.npz#T2", 0.45),        # T2
    ("SNAP:cnet2:/root/carbon/data/weights/cnet2_m9xj260g1.npz#T2", 0.35),        # T3
    ("SNAP:cnet2:/root/carbon/data/weights/cnet2_m7bsv2r1.npz#T2", 0.425),        # T2
    ("SNAP:cnet2:/root/carbon/data/weights/cnet2_m7bsv2g1.npz#T2", 0.30),         # T3
    ("SNAP:cnet2:/root/carbon/data/weights/cnet2_caragor1_ep19.npz#T2", 0.275),   # T3
    ("SNAP:cnet2:/root/carbon/data/weights/cnet2_m9xj320r1.npz#T2", 0.55),        # T1
]


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


def last_row():
    try:
        with open(TRAIN, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 200000))
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
                args = [p.decode("utf-8", "replace") for p in f.read().split(b"\0") if p]
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
            if int(stat.rsplit(")", 1)[1].split()[2]) == pgid:
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
                if "spawn_main" in f.read().decode("utf-8", "replace"):
                    out.append(int(pid))
        except Exception:
            continue
    return out


def gpu_apps():
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    except Exception as e:
        return [f"ERR {e}"]


def gpu_used_mib():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def main():
    total = sum(w for _, w in POOL)
    log(f"WATCHER_START kl012+retier armed (base_it={BASE_IT}) pool_total={total}")
    assert abs(total - 5.5) < 1e-9
    pid, argv = find_trainer()
    if pid is None:
        log("ABORT no trainer found at arm time")
        return
    pgid = os.getpgid(pid)
    env = proc_env(pid)
    log(f"trainer pid={pid} pgid={pgid} nargs={len(argv)}")

    beat = time.time()
    while True:
        row = last_row()
        if row and row.get("it", 0) >= BASE_IT:
            log(f"it{BASE_IT} row seen kl={row.get('kl')} kl_coef={row.get('kl_coef')} wr={row.get('winrate')}")
            break
        if time.time() - beat > 300:
            log(f"waiting it{BASE_IT} (last_it={row.get('it') if row else None})")
            beat = time.time()
        time.sleep(20)

    # state stable + embedded it == BASE_IT
    prev = None
    embedded = None
    for _ in range(40):
        try:
            st = os.stat(STATE)
            sig = (st.st_size, st.st_mtime)
        except FileNotFoundError:
            sig = None
        if sig is not None and sig == prev:
            import torch
            embedded = int(torch.load(STATE, map_location="cpu", weights_only=False)["it"])
            break
        prev = sig
        time.sleep(30)
    log(f"state stable, embedded it={embedded}")
    if embedded != BASE_IT:
        log(f"ABORT embedded it {embedded} != {BASE_IT}")
        return

    subprocess.run(["cp", "-f", STATE, BACKUP_STATE], check=True)
    s, b = md5(STATE), md5(BACKUP_STATE)
    log(f"state backup {BACKUP_STATE} md5={b} match={s == b}")
    if s != b:
        log("ABORT state backup md5 mismatch")
        return

    members = group_members(pgid)
    log(f"group members before kill: {len(members)}")
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.time() + 120
    while time.time() < deadline and group_members(pgid):
        time.sleep(5)
    if group_members(pgid):
        log("SIGTERM incomplete, SIGKILL")
        os.killpg(pgid, signal.SIGKILL)
        deadline = time.time() + 60
        while time.time() < deadline and group_members(pgid):
            time.sleep(5)
    left, spawns, apps = group_members(pgid), spawn_main_pids(), gpu_apps()
    used = gpu_used_mib()
    log(f"post-kill checks: group_left={left} spawn_main={spawns} gpu_apps={apps} gpu_used_mib={used}")
    if left or spawns or apps:
        log("ABORT_DIRTY acceptance checks failed; NOT relaunching")
        return
    log("acceptance checks PASS")

    # modify exactly two argv values
    new_argv = list(argv)
    for flag, val in (("--kl", "0.12"), ("--pool", json.dumps([[n, w] for n, w in POOL]))):
        i = new_argv.index(flag)
        log(f"argv change {flag}: {new_argv[i+1][:60]}... -> {val[:60]}...")
        new_argv[i + 1] = val
    logf = open(LOG, "ab")
    p = subprocess.Popen(new_argv, cwd=CODE, env=env, stdout=logf,
                         stderr=subprocess.STDOUT, start_new_session=True)
    log(f"RELAUNCHED pid={p.pid} pgid={os.getpgid(p.pid)}")
    for delay in (30, 90):
        time.sleep(delay)
        try:
            os.kill(p.pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        log(f"postcheck +{delay}s alive={alive} gpu_used_mib={gpu_used_mib()}")
        if not alive:
            log("RELAUNCH_FAILED; NOT retrying")
            return
    try:
        with open(os.path.join(RUN, "effective_params.json")) as f:
            ep = json.load(f)
        log("effective_params " + json.dumps({k: ep.get(k) for k in
            ("it0", "kl_coef", "reward", "gamma", "rscale", "value_head")}, ensure_ascii=False))
    except Exception as e:
        log(f"effective_params read failed: {e}")

    beat = time.time()
    while True:
        row = last_row()
        if row and row.get("it", 0) >= BASE_IT + 1:
            log(f"it{BASE_IT+1} row kl={row.get('kl')} kl_coef={row.get('kl_coef')} wr={row.get('winrate')} warm={row.get('warm')}")
            break
        if time.time() - beat > 300:
            log(f"waiting it{BASE_IT+1} (last_it={row.get('it') if row else None})")
            beat = time.time()
        time.sleep(60)
    log("SWAP_COMPLETE")


if __name__ == "__main__":
    main()
