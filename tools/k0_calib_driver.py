#!/usr/bin/env python3
"""K0 round-2 SS6.4 calibration sweep driver (DGX line, 2026-11-15).

Frozen checklist (strategy-group ruling, quoted in the calibration receipt):
  seg1: kl-beta 1.0 fixed, (lr, steps) in {2e-4,5e-4,1e-3} x {100,200,400},
        ascending lr*steps, bs 256, arm C only (--flatten-q), seed fixed 7.
  neutral: duel cluster_bootstrap_ci95 (per-game clusters) contains 0.
Resumable: cells whose duel JSON exists are skipped.
"""
import json, os, subprocess, sys, time, hashlib

PY = "/home/arm/junyang/carbon/.venv/bin/python"
CODE = "/home/arm/junyang/carbon-repo/code"
DATA = "/home/arm/junyang/carbon/data/k0_r2_shard1_ds"
PARENT = "/home/arm/junyang/k0_dist/pkg/weights/cnet2_m3s2r1_heads_v2.npz"
HOLD = "/home/arm/junyang/carbon/data/k0_calib_holdout"
BASE = "/home/arm/junyang/k0_calib"
CELLS_DIR = os.path.join(BASE, "cells")
DUELS_DIR = os.path.join(BASE, "duels")
MANIFEST = os.path.join(BASE, "roots_calib.json")
LEDGER = os.path.join(BASE, "ledger.jsonl")
os.makedirs(CELLS_DIR, exist_ok=True)
os.makedirs(DUELS_DIR, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def run(cmd, logf):
    log("RUN", " ".join(cmd))
    with open(logf, "a") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    log("rc=", r.returncode, " ".join(cmd[:2]))
    return r.returncode


def wait_production():
    for _ in range(240):
        n = 0
        rp = os.path.join(HOLD, "replays")
        if os.path.isdir(rp):
            n = len([f for f in os.listdir(rp) if f.endswith(".json.gz")])
        alive = subprocess.run(["pgrep", "-f", "selfplay_gen"],
                               capture_output=True, text=True).stdout.strip()
        log(f"production: replays={n} procs={'yes' if alive else 'no'}")
        if n >= 50 and not alive:
            return True
        time.sleep(15)
    return False


def build_manifest():
    rp = os.path.join(HOLD, "replays")
    files = sorted(f for f in os.listdir(rp) if f.endswith(".json.gz"))
    assert len(files) == 50, f"expected 50 replays, got {len(files)}"
    entries = []
    for f in files:
        game = f[: -len(".json.gz")]
        path = os.path.join(rp, f)
        for step in range(200, 299):
            for me in (0, 1):
                entries.append({"game": game, "path": path,
                                "step": step, "me": me})
    with open(MANIFEST, "w") as fh:
        json.dump(entries, fh)
    sha = hashlib.sha256(open(MANIFEST, "rb").read()).hexdigest()
    log(f"manifest: {len(entries)} roots sha256={sha}")
    return len(entries)


def cell_tag(lr, steps, beta, seed):
    return f"lr{lr:g}_s{steps}_b{beta:g}_seed{seed}"


def do_cell(lr, steps, beta, seed):
    tag = cell_tag(lr, steps, beta, seed)
    npz = os.path.join(CELLS_DIR, tag + ".npz")
    out = os.path.join(DUELS_DIR, tag + ".json")
    if os.path.exists(out):
        log("skip (duel exists)", tag)
        return json.load(open(out))["summary"]
    if not os.path.exists(npz):
        rc = run([PY, os.path.join(CODE, "k0_distill.py"),
                  "--data", DATA, "--parent", PARENT, "--out", npz,
                  "--flatten-q", "--kl-beta", str(beta),
                  "--lr", str(lr), "--bs", "256",
                  "--max-updates", str(steps), "--seed", str(seed),
                  "--device", "cuda"],
                 os.path.join(BASE, f"train_{tag}.log"))
        if rc != 0 or not os.path.exists(npz):
            log("TRAIN FAIL", tag)
            return None
    rc = run([PY, os.path.join(CODE, "k0_duel.py"),
              "--student", npz, "--parent", PARENT,
              "--roots", MANIFEST, "--out", out,
              "--step-min", "200", "--step-max", "299",
              "--seed-cont", "995400", "--teacher-style", "2",
              "--workers", "20"],
             os.path.join(BASE, f"duel_{tag}.log"))
    if rc != 0 or not os.path.exists(out):
        log("DUEL FAIL", tag)
        return None
    s = json.load(open(out))["summary"]
    ci = s["cluster_bootstrap_ci95"]
    rec = {"tag": tag, "lr": lr, "steps": steps, "beta": beta, "seed": seed,
           "delta_u": s["delta_u"], "ci95": ci, "n_games": s["n_games"],
           "n_roots": s["n_roots"], "neutral": bool(ci[0] <= 0 <= ci[1])}
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    log("CELL DONE", json.dumps(rec))
    return s


def main():
    assert hashlib.md5(open(PARENT, "rb").read()).hexdigest() == \
        "06a03d9cfeec239af3065a49fefec382", "parent md5 mismatch"
    if not wait_production():
        log("production did not complete in time; aborting")
        sys.exit(3)
    build_manifest()
    cells = [(2e-4, 100), (2e-4, 200), (5e-4, 100), (2e-4, 400),
             (5e-4, 200), (1e-3, 100), (5e-4, 400), (1e-3, 200),
             (1e-3, 400)]
    for lr, steps in cells:
        do_cell(lr, steps, 1.0, 7)
    log("SEG1 COMPLETE")


if __name__ == "__main__":
    main()
