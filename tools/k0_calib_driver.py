#!/usr/bin/env python3
"""K0 round-2 SS6.4 calibration sweep driver (DGX line, 2026-11-15).

v2 (2026-11-15 ~18:1x): sharded duels per strategy-group order (owner 13:41
rule item 4). k0_duel writes its JSON only at completion, so a single full
duel is a ~4h zero-intermediate-output black box. v2 runs the duel as 50
per-game shards (198 roots each, same k0_duel binary, same args, same
per-root seeds), landing one shard JSON + updating a progress file after
each game, then merges with merge_duel_shards.py (validated bit-identical
to a full run on the cell-4 golden: summary, cluster CI, bootstrap CI,
rows and roots_sha256 all exactly equal).

Frozen checklist (strategy-group ruling, quoted in the calibration receipt):
  seg1: kl-beta 1.0 fixed, (lr, steps) in {2e-4,5e-4,1e-3} x {100,200,400},
        ascending lr*steps, bs 256, arm C only (--flatten-q), seed fixed 7.
  neutral: duel cluster_bootstrap_ci95 (per-game clusters) contains 0.
Resumable: cells whose duel JSON exists are skipped; shards whose JSON
exists are skipped; progress file resumes.
"""
import json, os, subprocess, sys, time, hashlib, datetime

PY = "/home/arm/junyang/carbon/.venv/bin/python"
CODE = "/home/arm/junyang/carbon-repo/code"
DATA = "/home/arm/junyang/carbon/data/k0_r2_shard1_ds"
PARENT = "/home/arm/junyang/k0_dist/pkg/weights/cnet2_m3s2r1_heads_v2.npz"
HOLD = "/home/arm/junyang/carbon/data/k0_calib_holdout"
BASE = "/home/arm/junyang/k0_calib"
CELLS_DIR = os.path.join(BASE, "cells")
DUELS_DIR = os.path.join(BASE, "duels")
SHARDS_DIR = os.path.join(BASE, "shards")
MANIFEST = os.path.join(BASE, "roots_calib.json")
LEDGER = os.path.join(BASE, "ledger.jsonl")
MERGE = os.path.join(BASE, "merge_duel_shards.py")
BUILD_SHARDS = os.path.join(BASE, "build_duel_shards.py")
os.makedirs(CELLS_DIR, exist_ok=True)
os.makedirs(DUELS_DIR, exist_ok=True)
os.makedirs(SHARDS_DIR, exist_ok=True)


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


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


def write_progress(tag, done_games, total_games, done_roots, total_roots,
                   done_ids, started_ts, last_game):
    elapsed = time.time() - started_ts
    rate = (done_roots / elapsed) if elapsed > 0 and done_roots else None
    eta = None
    if rate:
        remain_s = (total_roots - done_roots) / rate
        eta = (datetime.datetime.now(datetime.timezone.utc)
               + datetime.timedelta(seconds=remain_s)).strftime("%Y-%m-%d %H:%M:%S UTC")
    prog = {"tag": tag, "done_games": done_games, "total_games": total_games,
            "done_roots": done_roots, "total_roots": total_roots,
            "done_game_ids": done_ids, "last_shard_game": last_game,
            "started_utc": datetime.datetime.fromtimestamp(
                started_ts, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "last_write_utc": utcnow(),
            "elapsed_s": round(elapsed, 1),
            "est_finish_utc": eta}
    with open(os.path.join(BASE, f"progress_{tag}.json"), "w") as fh:
        json.dump(prog, fh, indent=1)
    return prog


def run_sharded_duel(tag, npz, out, logf):
    """Duel as 50 per-game shards + merge. Returns merged summary or None."""
    shard_dir = os.path.join(SHARDS_DIR, tag)
    rc = run([PY, BUILD_SHARDS, tag], logf)
    if rc != 0:
        log("SHARD BUILD FAIL", tag)
        return None
    index = json.load(open(os.path.join(shard_dir, "index.json")))
    games = index["games"]
    total_games = len(games)
    total_roots = index["total_roots"]
    started_ts = time.time()
    done_ids = []
    done_roots = 0
    shard_paths = []
    for i, g in enumerate(games):
        shard_manifest = os.path.join(shard_dir, f"game-{g}.json")
        shard_out = os.path.join(shard_dir, f"duel-{g}.json")
        shard_paths.append(shard_out)
        if os.path.exists(shard_out):
            log(f"SHARD skip (exists) {i+1}/{total_games} {g}")
            done_ids.append(g)
            done_roots += 198
            continue
        rc = run([PY, os.path.join(CODE, "k0_duel.py"),
                  "--student", npz, "--parent", PARENT,
                  "--roots", shard_manifest, "--out", shard_out,
                  "--step-min", "200", "--step-max", "299",
                  "--seed-cont", "995400", "--teacher-style", "2",
                  "--workers", "20"], logf)
        if rc != 0 or not os.path.exists(shard_out):
            log("SHARD DUEL FAIL", tag, g)
            return None
        done_ids.append(g)
        done_roots += 198
        prog = write_progress(tag, len(done_ids), total_games, done_roots,
                              total_roots, done_ids, started_ts, g)
        log(f"SHARD {len(done_ids)}/{total_games} {g} done_roots={done_roots} "
            f"est_finish={prog['est_finish_utc']}")
    rc = run([PY, MERGE, MANIFEST] + shard_paths + [out], logf)
    if rc != 0 or not os.path.exists(out):
        log("MERGE FAIL", tag)
        return None
    s = json.load(open(out))["summary"]
    write_progress(tag, total_games, total_games, total_roots, total_roots,
                   done_ids, started_ts, games[-1])
    log("DUEL MERGED", tag, json.dumps(s))
    return s


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
    s = run_sharded_duel(tag, npz, out, os.path.join(BASE, f"duel_{tag}.log"))
    if s is None:
        log("DUEL FAIL", tag)
        return None
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
