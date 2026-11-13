#!/usr/bin/env python3
"""distill_eval -- Case-B stage 1 fidelity gates (design section 1.9).

Gate games (paired, FastCarbon, eval seeding protocol
random.seed(7919*seed + 104729*policy_side + 31), full traces):
  pure    distilled_v1.py         vs base_v53, seeds 1000-1003 x2 = 8
  hybrid  distilled_v1_hybrid.py  vs base_v53, seeds 1000-1003 x2 = 8

Stress games (opening double-kill, shadow-DS lesson): at step 40,
before either side acts, ALL collectors of the distilled side are
deleted from the sim (fastsim _kill_worker bookkeeping replicated).
  pure/hybrid x seeds 1000-1001 x2 sides = 4 each.
Tracked per stress game: first step a new collector exists again,
final cash, whether the economy restarted (recruit + cash >= 1500).

Verification: after the batch, 3 recorded games (1 pure, 1 hybrid,
1 stress) are re-run in-process and their final cash must match the
recorded values exactly (0.00 check).

Outputs: distill_eval_results.jsonl, traces/*.json.gz + INDEX rows.
At most 2 games concurrently.
"""

import importlib.util
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))

import record_trace  # noqa: E402

RESULTS = os.path.join(HERE, "distill_eval_results.jsonl")
BASE = "base_v53.py"
VARIANTS = [("pure", "distilled_v1.py"), ("hybrid", "distilled_v1_hybrid.py")]
STRESS_STEP = 40

_counter = [0]


def load_agent_fresh(path):
    _counter[0] += 1
    name = f"deval_{_counter[0]}_{os.getpid()}_" + os.path.splitext(
        os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "make_agent"):
        return mod.make_agent()
    return mod.agent


def kill_collectors(sim, player):
    """Delete all collectors of `player` (mirrors fastsim._kill_worker
    bookkeeping minus the step-local cell_worker map)."""
    killed = 0
    for wid in list(sim.player_workers[player]):
        w = sim.workers[wid]
        if w.is_collector:
            sim.player_workers[player].remove(wid)
            del sim.workers[wid]
            for entry in sim.trees_dict.values():
                if entry[0] == wid:
                    entry[0] = None
            killed += 1
    return killed


def play(job):
    variant, vfile, seed, side, stress = job
    random.seed(7919 * seed + 104729 * side + 31)
    sim = record_trace.FastCarbon(record_trace.CFG).reset(seed=seed)
    policy = load_agent_fresh(os.path.join(HERE, vfile))
    opp = load_agent_fresh(os.path.join(HERE, BASE))
    agents = [None, None]
    agents[side] = policy
    agents[1 - side] = opp

    frame0, st0 = record_trace.snapshot(sim)
    bases = {str(p): [[bid, pos] for bid, pos in st0["bases"][p]]
             for p in range(2)}
    frames = [frame0]
    killed = 0
    cash_at_kill = None
    first_rec_step = None
    for k in range(300):
        if stress and k == STRESS_STEP:
            cash_at_kill = float(sim.cash[side])
            killed = kill_collectors(sim, side)
        acts = []
        for p in range(2):
            o = sim.obs(p)
            acts.append(agents[p](o, record_trace.CFG))
        sim.step(acts[0], acts[1])
        if stress and k >= STRESS_STEP and first_rec_step is None:
            n_coll = sum(1 for wid in sim.player_workers[side]
                         if sim.workers[wid].is_collector)
            if n_coll > 0:
                first_rec_step = k + 1
        frame, _ = record_trace.snapshot(sim)
        frame["acts"] = [
            {kk: vv for kk, vv in (a or {}).items() if vv is not None}
            for a in acts]
        frames.append(frame)

    final_cash = frames[-1]["cash"]
    pol_stem = os.path.splitext(os.path.basename(vfile))[0]
    tag = f"distill_{variant}" + ("_stress" if stress else "")
    fname = f"{pol_stem}_vs_base_v53_seed{seed}_side{side}" + \
        ("_stress" if stress else "") + ".json.gz"
    trace = {
        "meta": {
            "format": "carbon-trace/1",
            "policy": pol_stem, "opp": "base_v53", "tag": tag,
            "seed": seed, "policy_side": side, "steps": 300,
            "size": int(record_trace.CFG["size"]),
            "bases": bases, "final_cash": final_cash,
            "stress": ({"kill_step": STRESS_STEP, "killed": killed,
                        "cash_at_kill": cash_at_kill}
                       if stress else None),
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
    my_cash = float(final_cash[side])
    op_cash = float(final_cash[1 - side])
    return {
        "tag": tag, "variant": variant, "stress": bool(stress),
        "seed": seed, "side": side, "file": fname,
        "final_cash": [float(final_cash[0]), float(final_cash[1])],
        "my_cash": my_cash, "opp_cash": op_cash,
        "margin": my_cash - op_cash,
        "annihilated": bool(my_cash < 500 and (my_cash - op_cash) < -6000),
        "killed": killed, "cash_at_kill": cash_at_kill,
        "first_rec_step": first_rec_step,
    }


def _job(job):
    t0 = time.time()
    last_err = None
    for attempt in (1, 2):
        try:
            r = play(job)
            r["attempts"] = attempt
            r["ok"] = True
            r["sec"] = round(time.time() - t0, 1)
            return r
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
    return {"tag": job[0], "seed": job[2], "side": job[3],
            "stress": bool(job[4]), "ok": False, "attempts": 2,
            "error": last_err, "sec": round(time.time() - t0, 1)}


def all_jobs(variants=None, main_seeds=(1000, 1001, 1002, 1003),
             stress_seeds=(1000, 1001)):
    jobs = []
    for variant, vfile in (variants or VARIANTS):
        for seed in main_seeds:
            for side in (0, 1):
                jobs.append((variant, vfile, seed, side, False))
        for seed in stress_seeds:
            for side in (0, 1):
                jobs.append((variant, vfile, seed, side, True))
    return jobs


def main():
    global RESULTS
    variants = VARIANTS
    main_seeds = (1000, 1001, 1002, 1003)
    if "--mini" in sys.argv:
        # Checkpoint-1 smoke gate (small-steps directive): pure variant
        # only, seeds 1000-1001 paired + stress pair, own output files.
        variants = [("pure", "smoke/distilled_v0.py")]
        main_seeds = (1000, 1001)
        RESULTS = os.path.join(HERE, "distill_eval_mini_results.jsonl")
    if "--mini1" in sys.argv:
        # Segment-2 mini re-check with the production weights: both
        # variants, seeds 1000-1001 paired + stress, own output files.
        main_seeds = (1000, 1001)
        RESULTS = os.path.join(HERE, "distill_eval_mini1_results.jsonl")
    jobs = all_jobs(variants, main_seeds)
    print(f"eval jobs={len(jobs)}", flush=True)
    results = []
    nworkers = 4 if "--workers4" in sys.argv else 2
    with ProcessPoolExecutor(max_workers=nworkers) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if r["ok"]:
                print(f"OK  {r['tag']} seed{r['seed']} side{r['side']} "
                      f"cash={r['final_cash']} margin={r['margin']:.0f} "
                      f"({r['sec']}s)", flush=True)
            else:
                print(f"FAIL {r.get('tag')} {r.get('error')}", flush=True)
    with open(RESULTS, "w") as f:
        for r in sorted(results, key=lambda x: (x.get("tag", ""),
                                                x.get("seed", 0),
                                                x.get("side", 0))):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 0.00 rerun verification (fresh agents, in-process) ----
    ok_rows = [r for r in results if r["ok"]]
    picks = []
    for want_tag in ("distill_pure", "distill_hybrid",
                     "distill_hybrid_stress", "distill_pure_stress"):
        cand = [r for r in ok_rows if r["tag"] == want_tag]
        if cand:
            picks.append(cand[0])
    verify = []
    for r in picks:
        variant = r["variant"]
        vfile = dict(variants)[variant]
        job = (variant, vfile, r["seed"], r["side"], r["stress"])
        rr = play(job)
        match = rr["final_cash"] == r["final_cash"]
        verify.append({"tag": r["tag"], "seed": r["seed"],
                       "side": r["side"],
                       "recorded": r["final_cash"],
                       "rerun": rr["final_cash"], "match": match})
        print(f"VERIFY {r['tag']} seed{r['seed']} side{r['side']}: "
              f"{'MATCH' if match else 'MISMATCH ' + str(rr['final_cash'])}",
              flush=True)
    vname = ("distill_eval_verify_mini.json" if "--mini" in sys.argv
             else "distill_eval_verify_mini1.json"
             if "--mini1" in sys.argv else "distill_eval_verify.json")
    with open(os.path.join(HERE, vname), "w") as f:
        json.dump(verify, f, ensure_ascii=False, indent=1)
    bad = [v for v in verify if not v["match"]]
    print(f"verify: {len(verify) - len(bad)}/{len(verify)} match")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
