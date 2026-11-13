#!/usr/bin/env python3
"""dagger_rollout -- Case-B segment 2: DAgger correction data.

Checkpoint 1 showed textbook compounding error: the distilled
student drifts into states the teacher never visits (2-worker
rosters, cash in the 20s) and has no supervision there.  This
rollout plays the STUDENT (smoke distilled_v0, pure) against
base_v53 while a GHOST TEACHER (a base_v53 TeacherRecorder on an
isolated random stream) labels every visited state with the
teacher's action.  The recorded trace's state evolution comes from
the student's actions; the student side's recorded acts are the
TEACHER's labels, so the stock extract/build pipeline turns the
traces into training samples in exactly the corpus schema.

Seeds 3000-3003, 3005, 3006 (none = 4 mod 5, so the build's
seed%5==4 validation split keeps every DAgger game in TRAIN).
12 games (x2 sides), <=2 concurrent.  Outputs:
  traces/distilled_v0_dagger_vs_base_v53_seed*_side*.json.gz
  dagger_manifest.json, bc_data/dagger_capture.npz,
  dagger_rollout_results.jsonl
"""

import gzip
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src",
                                "carbon_challenge-master"))

import numpy as np  # noqa: E402
import record_trace  # noqa: E402
import distill_capture as DC  # noqa: E402
import distill_eval as DE  # noqa: E402

SEEDS = (3000, 3001, 3002, 3003, 3005, 3006)
STUDENT = "smoke/distilled_v0.py"
TAG = "dagger"
if "--student" in sys.argv:
    STUDENT = sys.argv[sys.argv.index("--student") + 1]
if "--seeds" in sys.argv:
    SEEDS = tuple(int(x) for x in
                  sys.argv[sys.argv.index("--seeds") + 1].split(","))
if "--tag" in sys.argv:
    TAG = sys.argv[sys.argv.index("--tag") + 1]
TEACHER = "base_v53.py"


def play(job):
    seed, side = job
    random.seed(7919 * seed + 104729 * side + 31)
    sim = record_trace.FastCarbon(record_trace.CFG).reset(seed=seed)
    student = DE.load_agent_fresh(os.path.join(HERE, STUDENT))
    opp = DE.load_agent_fresh(os.path.join(HERE, TEACHER))
    tmod = DC.load_module_fresh(os.path.join(HERE, TEACHER))
    ghost = DC.TeacherRecorder(tmod, side)
    ghost_state = random.Random(987654 + seed * 100 + side).getstate()
    agents = [None, None]
    agents[side] = student
    agents[1 - side] = opp

    frame0, st0 = record_trace.snapshot(sim)
    bases = {str(p): [[bid, pos] for bid, pos in st0["bases"][p]]
             for p in range(2)}
    frames = [frame0]
    for _k in range(300):
        obs = [sim.obs(0), sim.obs(1)]
        game_state = random.getstate()
        random.setstate(ghost_state)
        t_acts = ghost(obs[side], record_trace.CFG)
        ghost_state = random.getstate()
        random.setstate(game_state)
        acts = [None, None]
        acts[side] = agents[side](obs[side], record_trace.CFG)
        acts[1 - side] = agents[1 - side](obs[1 - side],
                                          record_trace.CFG)
        sim.step(acts[0], acts[1])
        frame, _ = record_trace.snapshot(sim)
        rec_acts = [{kk: vv for kk, vv in (a or {}).items()
                     if vv is not None} for a in acts]
        rec_acts[side] = {kk: vv for kk, vv in (t_acts or {}).items()
                          if vv is not None}
        frame["acts"] = rec_acts
        frames.append(frame)

    final_cash = frames[-1]["cash"]
    stem = os.path.splitext(os.path.basename(STUDENT))[0]
    fname = f"{stem}_dagger_vs_base_v53_seed{seed}_side{side}" \
            ".json.gz"
    trace = {
        "meta": {
            "format": "carbon-trace/1",
            "policy": f"{stem}_dagger", "opp": "base_v53",
            "tag": TAG, "seed": seed, "policy_side": side,
            "steps": 300, "size": int(record_trace.CFG["size"]),
            "bases": bases, "final_cash": final_cash,
            "stress": None,
            "note": "states driven by distilled_v0 (student); the "
                    "recorded acts of the policy side are ghost-"
                    "teacher (base_v53) labels for DAgger training",
            "legend": {
                "workers": "wid -> [pos, cargo, C=COLLECTOR|P=PLANTER]",
                "trees": "tid -> [pos, owner_player, age]",
                "pos_index": "row * size + col (obs index space)",
                "cash": "cash IS the score in this game",
            },
        },
        "frames": frames,
    }
    path = os.path.join(record_trace.TRACES_DIR, fname)
    os.makedirs(record_trace.TRACES_DIR, exist_ok=True)
    record_trace.write_trace(trace, path)
    rows = sorted(ghost.rows, key=lambda r: r["step"])
    return {
        "seed": seed, "side": side, "file": fname,
        "final_cash": [float(final_cash[0]), float(final_cash[1])],
        "ghost_errors": ghost.errors,
        "belief": np.stack([r["belief"] for r in rows]),
        "known": np.stack([r["known"] for r in rows]),
        "prev_act": np.stack([r["prev_act"] for r in rows]),
        "seq": np.stack([r["seq"] for r in rows]),
        "since_dep": np.asarray([r["since_dep"] for r in rows],
                                dtype=np.float32),
        "steps": np.asarray([r["step"] for r in rows],
                            dtype=np.int16),
    }


def _job(job):
    t0 = time.time()
    try:
        r = play(job)
        r["ok"] = True
    except Exception as e:  # noqa: BLE001
        r = {"seed": job[0], "side": job[1], "ok": False,
             "error": f"{type(e).__name__}: {e}"}
    r["sec"] = round(time.time() - t0, 1)
    return r


def main():
    jobs = [(s, sd) for s in SEEDS for sd in (0, 1)]
    print(f"dagger jobs={len(jobs)}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if r["ok"]:
                print(f"OK seed{r['seed']} side{r['side']} "
                      f"cash={r['final_cash']} ({r['sec']}s)",
                      flush=True)
            else:
                print(f"FAIL seed{r['seed']} side{r['side']} "
                      f"{r['error']}", flush=True)
    ok = [r for r in results if r["ok"]]
    assert len(ok) == len(jobs), "dagger rollout incomplete"
    manifest = [{
        "file": r["file"], "tag": TAG, "seed": r["seed"],
        "side": r["side"],
        "names": {"0": "base_v53", "1": "base_v53"},
        "teacher_players": [r["side"]],
        "final_cash": r["final_cash"],
    } for r in sorted(ok, key=lambda x: x["file"])]
    with open(os.path.join(HERE, f"{TAG}_manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    files = []
    rows = {"file_idx": [], "player": [], "step": [], "belief": [],
            "known": [], "prev_act": [], "seq": [], "since_dep": []}
    total_errors = 0
    for r in sorted(ok, key=lambda x: x["file"]):
        fidx = len(files)
        files.append(r["file"])
        total_errors += int(r["ghost_errors"])
        n = len(r["steps"])
        rows["file_idx"].append(np.full(n, fidx, dtype=np.int32))
        rows["player"].append(np.full(n, r["side"], dtype=np.int8))
        rows["step"].append(r["steps"])
        rows["belief"].append(r["belief"])
        rows["known"].append(r["known"])
        rows["prev_act"].append(r["prev_act"])
        rows["seq"].append(r["seq"])
        rows["since_dep"].append(r["since_dep"])
    packed = {k: np.concatenate(v) for k, v in rows.items()}
    packed["files"] = np.asarray(files)
    out = os.path.join(HERE, "bc_data", f"{TAG}_capture.npz")
    np.savez_compressed(out, **packed)
    print(f"wrote {out}: rows={len(packed['step'])} "
          f"ghost_errors={total_errors}")
    with open(os.path.join(HERE, f"{TAG}_rollout_results.jsonl"),
              "w") as f:
        for r in sorted(ok, key=lambda x: x["file"]):
            f.write(json.dumps(
                {"file": r["file"], "seed": r["seed"],
                 "side": r["side"], "final_cash": r["final_cash"],
                 "ghost_errors": r["ghost_errors"]}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
