#!/usr/bin/env python3
"""ppou2 dual-arm launcher (list-argv, own process group, launch record).

Usage: launch_ppou2.py m|c
Arms share everything except --lam and --out (see FILING.md).
"""
import json, os, subprocess, sys, time

ARM = sys.argv[1]
assert ARM in ("m", "c")
LAM = "0.95" if ARM == "m" else "1.0"
OUT = f"/root/carbon/data/rl3/ppou2_{ARM}"
LOG = f"/root/carbon/data/rl3/ppou2_{ARM}.log"
W = "/root/carbon/data/weights"
POOL = [
    [f"SNAP:cnet2:{W}/cnet2_m3s2r1L.npz#T2", 1.10],
    [f"SNAP:cnet2:{W}/cnet2_m3s2g1L.npz#T2", 0.55],
    [f"SNAP:cnet2:{W}/cnet2_m7bsr1L.npz#T2", 0.55],
    [f"SNAP:cnet2:{W}/cnet2_m9xj260wr1_fromstudio.npz#T2", 0.50],
    [f"SNAP:cnet2:{W}/cnet2_m9xj260r1.npz#T2", 0.45],
    [f"SNAP:cnet2:{W}/cnet2_m6r1L.npz#T2", 0.45],
    [f"SNAP:cnet2:{W}/cnet2_m7bsv2r1.npz#T2", 0.425],
    [f"SNAP:cnet2:{W}/cnet2_m9xj260g1.npz#T2", 0.35],
    [f"SNAP:cnet2:{W}/cnet2_m7bsv2g1.npz#T2", 0.30],
    [f"SNAP:cnet2:{W}/cnet2_caragor1_ep19.npz#T2", 0.275],
]
argv = [
    "/root/carbon/.venv/bin/python", "rl2_ppo_train.py",
    "--adapter", "cnet2", "--device", "cuda", "--server", "1",
    "--reward", "u", "--value-head", "tanh",
    "--init", f"{W}/cnet2_m3s2r1L.npz",
    "--pool-authoritative", "--pool", json.dumps(POOL),
    "--self-prior", "0.5", "--snap-prior", "1", "--snap-every", "10", "--max-snaps", "2",
    "--vwarm", "3", "--workers", "14", "--lockstep", "16",
    "--dec-frac", "0.25", "--epochs", "1", "--mb", "768", "--pre-mb", "4096",
    "--games", "4000", "--lr", "1e-5", "--kl", "0.20",
    "--gamma", "1.0", "--rscale", "1.0",
    "--ckpt-every", "1", "--eval-every", "5", "--eval-pairs", "32",
    "--eval-opps", "exp/opp_master2r.py,exp/opp_m3echo.py",
    "--seed0", "3000000", "--lam", LAM, "--out", OUT,
]
env = dict(os.environ)
env.update({
    "CNET2_TEACHER": "2", "RL2_SNAP_TEMP": "0.5", "CNET2_TEMP_KIND": "0.15,0.3,0.3",
    "RL2_SERVERS": "1", "RL2_SERVER_MAXB": "1024",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
})
os.makedirs(OUT, exist_ok=True)
logf = open(LOG, "ab")
p = subprocess.Popen(argv, cwd="/root/carbon/code", env=env,
                     stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
rec = {"arm": ARM, "lam": float(LAM), "pid": p.pid, "pgid": os.getpgid(p.pid),
       "out": OUT, "log": LOG, "argv": argv, "started": time.strftime("%F %T %Z"),
       "expected_workers": 14}
with open(f"/root/carbon/data/rl3/ppou2_{ARM}.launch.json", "w") as f:
    json.dump(rec, f, indent=1)
print(json.dumps(rec))
