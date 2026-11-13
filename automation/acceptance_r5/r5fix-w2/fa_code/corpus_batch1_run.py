#!/usr/bin/env python3
"""corpus_batch1_run: play + record the corpus-batch-1 game matrix.

Batch purpose (2026-11-03): v53 self-play / sparring corpus for the
Plan-B distillation (v53 as teacher) plus lesion-analysis material.
Matrix (120 games, seeds 2000-2059, sides balanced):
  cb1_mirror  base_v53 vs base_v53                    40 (seeds 2000-2019)
  cb1_ge      base_v53 vs GloriousEvolution-v53       20 (seeds 2020-2029)
  cb1_shadow  base_v53 vs shadow_alpha_v2             20 (seeds 2030-2039)
  cb1_v05     base_v53 vs policy_v0_5                 20 (seeds 2040-2049)
  cb1_fork    fork_v0_1 vs base_v53                   20 (seeds 2050-2059)

Recording: same loop order and trace format as record_trace.py
(reset -> load policy -> load opp -> per step obs/act p0 then p1),
same seeding protocol as eval_v53 / record_v53_traces:
  random.seed(7919*seed + 104729*policy_side + 31) before the game.
Every game is retried once on failure; failures are recorded, never
silently dropped.  At most 2 games run concurrently (ProcessPool of 2).

Outputs:
  traces/<pol>_vs_<opp>_seed<seed>_side<side>.json.gz   (per game)
  corpus_batch1_results.jsonl                           (one row / game)
  corpus_batch1_manifest.json                           (extraction input)
  corpus_batch1_obscheck.json  (only with --validate-obs: live obs
                                carbon at states 0/50/150/299 of the
                                first mirror game, for extractor checks)

Usage:
  python3 corpus_batch1_run.py [--only TAG] [--validate-obs]
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

import record_trace  # noqa: E402  (snapshot / write_trace / CFG / FastCarbon)

RESULTS = os.path.join(HERE, "corpus_batch1_results.jsonl")
MANIFEST = os.path.join(HERE, "corpus_batch1_manifest.json")

# (tag, policy_file, opp_file, policy_name, opp_name, seeds, teachers)
# teachers: which side(s) are v53-family teachers for BC extraction,
# relative to the record: "policy", "opp" or "both".
MATRIX = [
    ("cb1_mirror", "base_v53.py", "base_v53.py",
     "base_v53", "base_v53", range(2000, 2020), "both"),
    ("cb1_ge", "base_v53.py", "reference_gloriousevolution_v53.py",
     "base_v53", "GloriousEvolution-v53", range(2020, 2030), "policy"),
    ("cb1_shadow", "base_v53.py", "shadow_alpha_v2.py",
     "base_v53", "shadow_alpha_v2", range(2030, 2040), "policy"),
    ("cb1_v05", "base_v53.py", "policy_v0_5.py",
     "base_v53", "policy_v0_5", range(2040, 2050), "policy"),
    ("cb1_fork", "fork_v0_1.py", "base_v53.py",
     "fork_v0_1", "base_v53", range(2050, 2060), "both"),
]

_counter = [0]


def load_agent_fresh(path):
    """Fresh module instance per game (no cross-game state leakage);
    make_agent() when the file offers it, else module-level agent()."""
    _counter[0] += 1
    name = f"cb1_{_counter[0]}_{os.getpid()}_" + os.path.splitext(
        os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "make_agent"):
        return mod.make_agent()
    return mod.agent


def play_record(job, capture_obs_at=()):
    """One game, recorded exactly like record_trace.record_game but with
    the eval_v53 seeding protocol and a two-file loader."""
    seed, side = job["seed"], job["side"]
    random.seed(7919 * seed + 104729 * side + 31)
    sim = record_trace.FastCarbon(record_trace.CFG).reset(seed=seed)
    policy = load_agent_fresh(os.path.join(HERE, job["policy_file"]))
    opp = load_agent_fresh(os.path.join(HERE, job["opp_file"]))
    agents = [None, None]
    agents[side] = policy
    agents[1 - side] = opp

    frame0, st0 = record_trace.snapshot(sim)
    bases = {str(p): [[bid, pos] for bid, pos in st0["bases"][p]]
             for p in range(2)}
    frames = [frame0]
    obs_capture = {}
    for k in range(300):
        acts = []
        for p in range(2):
            o = sim.obs(p)
            if k in capture_obs_at and p == 0:
                obs_capture[k] = [None, None]
            if k in capture_obs_at:
                obs_capture[k][p] = list(o["carbon"])
            acts.append(agents[p](o, record_trace.CFG))
        sim.step(acts[0], acts[1])
        frame, _ = record_trace.snapshot(sim)
        frame["acts"] = [
            {kk: vv for kk, vv in (a or {}).items() if vv is not None}
            for a in acts]
        frames.append(frame)

    final_cash = frames[-1]["cash"]
    pol_stem = os.path.splitext(job["policy_file"])[0]
    opp_stem = os.path.splitext(job["opp_file"])[0]
    trace = {
        "meta": {
            "format": "carbon-trace/1",
            "policy": pol_stem,
            "opp": opp_stem,
            "tag": job["tag"],
            "seed": seed,
            "policy_side": side,
            "steps": 300,
            "size": int(record_trace.CFG["size"]),
            "bases": bases,
            "final_cash": final_cash,
            "legend": {
                "workers": "wid -> [pos_index, cargo, C=COLLECTOR|P=PLANTER]",
                "trees": "tid -> [pos_index, owner_player, age]",
                "pos_index": "row * size + col, row 0 = top (obs index "
                               "space)",
                "cash": "cash IS the score in this game",
            },
        },
        "frames": frames,
    }
    fname = f"{pol_stem}_vs_{opp_stem}_seed{seed}_side{side}.json.gz"
    path = os.path.join(record_trace.TRACES_DIR, fname)
    os.makedirs(record_trace.TRACES_DIR, exist_ok=True)
    raw_len, gz_len = record_trace.write_trace(trace, path)
    return {
        "tag": job["tag"], "seed": seed, "side": side,
        "policy_file": job["policy_file"], "opp_file": job["opp_file"],
        "policy_name": job["policy_name"], "opp_name": job["opp_name"],
        "teachers": job["teachers"],
        "final_cash": final_cash, "file": fname,
        "raw_bytes": raw_len, "gz_bytes": gz_len,
        "obs_capture": obs_capture,
    }


def _job(job):
    t0 = time.time()
    last_err = None
    for attempt in (1, 2):
        try:
            r = play_record(job)
            r["attempts"] = attempt
            r["ok"] = True
            r["sec"] = round(time.time() - t0, 1)
            return r
        except Exception as e:  # noqa: BLE001 - recorded, retried once
            last_err = f"{type(e).__name__}: {e}"
    return {"tag": job["tag"], "seed": job["seed"], "side": job["side"],
            "policy_file": job["policy_file"], "opp_file": job["opp_file"],
            "ok": False, "attempts": 2, "error": last_err,
            "sec": round(time.time() - t0, 1)}


def all_jobs(only=None):
    jobs = []
    for tag, pf, of, pn, on, seeds, teachers in MATRIX:
        if only and tag != only:
            continue
        for seed in seeds:
            for side in (0, 1):
                jobs.append({"tag": tag, "policy_file": pf, "opp_file": of,
                             "policy_name": pn, "opp_name": on,
                             "seed": seed, "side": side,
                             "teachers": teachers})
    return jobs


def done_keys():
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("ok"):
                    done.add((r["tag"], r["seed"], r["side"]))
    return done


def main():
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    validate_obs = "--validate-obs" in sys.argv

    jobs = all_jobs(only)
    done = done_keys()
    todo = [j for j in jobs
            if (j["tag"], j["seed"], j["side"]) not in done]
    print(f"jobs total={len(jobs)} already_done={len(jobs) - len(todo)} "
          f"todo={len(todo)}", flush=True)

    results = []
    if validate_obs and todo:
        # run the first job in-process with obs capture for extractor checks
        j0 = todo.pop(0)
        r = play_record(j0, capture_obs_at=(0, 50, 150, 299))
        cap = r.pop("obs_capture")
        with open(os.path.join(HERE, "corpus_batch1_obscheck.json"),
                  "w") as f:
            json.dump({"file": r["file"], "seed": j0["seed"],
                       "side": j0["side"],
                       "capture": {str(k): v for k, v in cap.items()}}, f)
        r.update({"attempts": 1, "ok": True, "sec": 0})
        results.append(r)
        print(f"[obscheck] {r['file']} cash={r['final_cash']}", flush=True)

    if todo:
        with ProcessPoolExecutor(max_workers=2) as ex:
            futs = {ex.submit(_job, j): j for j in todo}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                if r["ok"]:
                    print(f"OK  {r['tag']} seed{r['seed']} side{r['side']} "
                          f"cash={r['final_cash']} ({r['sec']}s)", flush=True)
                else:
                    print(f"FAIL {r['tag']} seed{r['seed']} side{r['side']} "
                          f"{r['error']}", flush=True)

    if results:
        with open(RESULTS, "a") as f:
            for r in sorted(results,
                            key=lambda x: (x["tag"], x["seed"], x["side"])):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # rebuild manifest from all OK results
    manifest = []
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not r.get("ok"):
                    continue
                names = {str(r["side"]): r["policy_name"],
                         str(1 - r["side"]): r["opp_name"]}
                if r["teachers"] == "both":
                    teacher_players = [0, 1]
                elif r["teachers"] == "policy":
                    teacher_players = [r["side"]]
                else:
                    teacher_players = [1 - r["side"]]
                manifest.append({
                    "file": r["file"], "tag": r["tag"], "seed": r["seed"],
                    "side": r["side"], "names": names,
                    "teacher_players": teacher_players,
                    "final_cash": r["final_cash"],
                })
    manifest.sort(key=lambda m: (m["tag"], m["seed"], m["side"]))
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    ok = sum(1 for m in manifest)
    print(f"manifest games={ok}")
    fails = [r for r in results if not r["ok"]]
    if fails:
        print(f"FAILURES this run: {len(fails)}")


if __name__ == "__main__":
    main()
