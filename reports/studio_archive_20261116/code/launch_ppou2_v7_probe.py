#!/usr/bin/env python3
"""ppou2 single-arm launcher v5 (list-argv, own process group, launch record).

v4 + --kl-value override (default 0.05 = the filed value). Used by the
serial driver's KL stop-line mechanism (strategy group ruling 2026-11-15
10:15: on any round with achieved KL >= 0.13, stop first, then resume with
kl_coef 0.15, both arms). Usage:
  launch_ppou2_v5.py m|c [--resume /path/to/state_last.pt]
                         [--kl-value 0.15] [--dry-run]
--dry-run prints the argv JSON and exits without launching.
Arms share everything except --lam and --out (see FILING.md).
"""
import json, os, subprocess, sys, time

ARM = sys.argv[1]
assert ARM in ("m", "c")
RESUME = None
REF = None
DRY = False
KL = "0.05"
_i = 2
while _i < len(sys.argv):
    if sys.argv[_i] == "--resume":
        _i += 1
        RESUME = os.path.abspath(sys.argv[_i])
    elif sys.argv[_i] == "--ref":
        _i += 1
        REF = os.path.abspath(sys.argv[_i])
    elif sys.argv[_i] == "--kl-value":
        _i += 1
        KL = sys.argv[_i]
    elif sys.argv[_i] == "--dry-run":
        DRY = True
    else:
        raise SystemExit(f"unknown arg: {sys.argv[_i]}")
    _i += 1
assert KL in ("0.05", "0.15"), f"kl-value not in ruled set: {KL}"
LAM = "0.95" if ARM == "m" else "1.0"
OUT = "/Users/arm/junyang/carbon/data/ppou2_probe_studio/out"
LOG = "/Users/arm/junyang/carbon/data/ppou2_probe_studio/out/ppou2_probe.log"
W = "/Users/arm/junyang/carbon/data/ppou2_probe_studio/weights"
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
    "/Users/arm/junyang/carbon/.venv/bin/python", "rl2_ppo_train.py",
    "--adapter", "cnet2", "--device", "mps", "--server", "1",
    "--reward", "u", "--value-head", "tanh",
    "--init", f"{W}/cnet2_m3s2r1L.npz",
    "--pool-authoritative", "--pool", json.dumps(POOL),
    "--self-prior", "0.5", "--snap-prior", "1", "--snap-every", "10", "--max-snaps", "2",
    "--vwarm", "3", "--workers", "14", "--lockstep", "16",
    "--dec-frac", "0.25", "--epochs", "1", "--mb", "768", "--pre-mb", "4096",
    "--games", "4000", "--lr", "1e-5", "--kl", KL,
    "--gamma", "1.0", "--rscale", "1.0",
    "--ckpt-every", "1", "--eval-every", "5", "--eval-pairs", "32",
    "--eval-opps", "exp/opp_master2r.py,exp/opp_m3echo.py",
    "--seed0", "3000000", "--lam", LAM, "--out", OUT,
]
if RESUME:
    argv += ["--resume", RESUME]
if REF:
    argv += ["--ref", REF]
if DRY:
    print(json.dumps({"arm": ARM, "resume": RESUME, "ref": REF, "kl": KL, "argv": argv}))
    sys.exit(0)
env = dict(os.environ)
env.update({
    "CNET2_TEACHER": "2", "RL2_SNAP_TEMP": "0.5", "CNET2_TEMP_KIND": "0.15,0.3,0.3",
    "RL2_SERVERS": "2", "RL2_SERVER_MAXB": "1024", "RL2_SERVER_MODELS": "20", "PYTORCH_ENABLE_MPS_FALLBACK": "1",
})
os.makedirs(OUT, exist_ok=True)
logf = open(LOG, "ab")
p = subprocess.Popen(argv, cwd="/Users/arm/junyang/carbon/data/ppou2_probe_studio/code", env=env,
                     stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
rec = {"arm": ARM, "lam": float(LAM), "pid": p.pid, "pgid": os.getpgid(p.pid),
       "out": OUT, "log": LOG, "argv": argv, "resume": RESUME, "ref": REF, "kl": KL, "servers": 2, "server_models": 20,
       "started": time.strftime("%F %T %Z"), "expected_workers": 14}
with open("/Users/arm/junyang/carbon/data/ppou2_probe_studio/out/ppou2_probe.launch.json", "w") as f:
    json.dump(rec, f, indent=1)
print(json.dumps(rec))
