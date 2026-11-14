#!/usr/bin/env python3
"""katago-probe probe-2 CONSTRUCTION GATE driver (PREREG v1.1).

Per registered root (game, seat, step in [200,299) of rootset32), drive a
FRESH search-agent instance per arm on the replay's recorded observation
and compare the chosen joint actions:

  value : variant module, conf LEAF_VALUE="value" (network tanh value leaf)
  const : variant module, conf LEAF_VALUE="const" (ablation, M_eq == 0 leaf)
  heur  : stock base module (hand-heuristic leaf) -- side reference only

Gate rule (PREREG): value-vs-const changed count > 0 == wiring live.

Driving convention (construction sanity only, no inferential reading):
each root is fed to a fresh agent instance as a single recorded
observation; the three arms see byte-identical obs (deep-copied) and the
same per-root RNG seed. Root obs assembly mirrors cnet2_build.build_game:
step/players/trees from player 0's full recorded observation, carbon =
the seat's own filtered view from its recorded entry.

Inputs:
  --replays DIR --rootset FILE   registered mode (replays = *.json.gz)
  --tasks FILE                   pre-assembled tasks JSONL (smoke only)
Outputs in --out: results.jsonl, summary.json, examples.json
"""
import argparse
import copy
import gzip
import hashlib
import json
import os
import sys
import time

SEED_BASE = 20261114
ARMS = ("value", "const", "heur")

G = {}   # per-worker globals


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def worker_init(moddir, weights_path, rweights_path):
    sys.path.insert(0, moddir)
    import numpy as np
    import cnet2_search as base_mod
    import cnet2_search_vleaf as vleaf_mod

    def load(p):
        d = np.load(p, allow_pickle=False)
        return {k: d[k] for k in d.files}

    G["base"] = base_mod
    G["vleaf"] = vleaf_mod
    G["w"] = load(weights_path)
    G["rw"] = load(rweights_path)
    G["np"] = np


def make_agent(arm):
    if arm == "heur":
        return G["base"].make_search_agent(conf=None, weights=G["w"],
                                           rweights=G["rw"])
    return G["vleaf"].make_search_agent(conf={"LEAF_VALUE": arm},
                                        weights=G["w"], rweights=G["rw"])


def canon(act):
    return json.dumps(act or {}, sort_keys=True)


def run_root(task):
    np = G["np"]
    idx = task["root_index"]
    rec = {"game_key": task["game_key"], "seat": task["seat"],
           "step": task["step"], "root_index": idx, "arms": {}}
    for arm in ARMS:
        agent = make_agent(arm)
        np.random.seed(SEED_BASE + idx)
        t0 = time.time()
        act = agent(copy.deepcopy(task["obs"]))
        dt = time.time() - t0
        st = agent.stats
        rec["arms"][arm] = {
            "action": canon(act),
            "searched": int(st.get("searched", 0)),
            "changed": int(st.get("changed", 0)),
            "branches": int(st.get("branches", 0)),
            "depth_sum": int(st.get("depth_sum", 0)),
            "errors": int(st.get("errors", 0)),
            "root_errors": int(st.get("root_errors", 0)),
            "dt_s": round(dt, 4),
        }
    rec["changed_value_const"] = rec["arms"]["value"]["action"] != rec["arms"]["const"]["action"]
    rec["changed_value_heur"] = rec["arms"]["value"]["action"] != rec["arms"]["heur"]["action"]
    rec["changed_const_heur"] = rec["arms"]["const"]["action"] != rec["arms"]["heur"]["action"]
    return rec


def tasks_from_replays(replays_dir, rootset_path):
    games = json.load(open(rootset_path))
    tasks = []
    idx = 0
    for g in games:
        path = os.path.join(replays_dir, g["key"])
        d = json.load(gzip.open(path, "rt"))
        steps = d["steps"]
        for seat in (0, 1):
            for t in range(200, 299):
                if steps[t][seat].get("status") not in (None, "ACTIVE"):
                    continue
                o0 = steps[t][0]["observation"]
                ep = steps[t][seat]["observation"]
                assert int(o0["step"]) == t, (g["key"], t, o0["step"])
                obs = {"step": int(o0["step"]), "players": o0["players"],
                       "carbon": ep["carbon"], "player": seat,
                       "trees": o0.get("trees") or {},
                       "remainingOverageTime": ep.get(
                           "remainingOverageTime",
                           o0.get("remainingOverageTime"))}
                tasks.append({"game_key": g["key"], "seat": seat, "step": t,
                              "root_index": idx, "obs": obs})
                idx += 1
    return tasks


def tasks_from_file(path):
    tasks = []
    for i, line in enumerate(open(path)):
        r = json.loads(line)
        r["root_index"] = i
        tasks.append(r)
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moddir", required=True)
    ap.add_argument("--replays")
    ap.add_argument("--rootset")
    ap.add_argument("--tasks")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--rweights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    if a.tasks:
        tasks = tasks_from_file(a.tasks)
        src_desc = {"tasks_file": a.tasks}
    else:
        tasks = tasks_from_replays(a.replays, a.rootset)
        src_desc = {"replays": a.replays, "rootset": a.rootset,
                    "rootset_md5": md5_file(a.rootset)}
    print(f"tasks: {len(tasks)} roots", flush=True)

    file_md5 = {}
    for name in ("cnet2_search_vleaf.py", "cnet2_search.py", "cnet2_feat.py",
                 "cnet2_np.py", "fastsim.py", "gate_driver.py"):
        p = os.path.join(a.moddir, name) if name != "gate_driver.py" else \
            os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        if os.path.exists(p):
            file_md5[name] = md5_file(p)
    file_md5["weights_npz"] = md5_file(a.weights)
    file_md5["rweights_npz"] = md5_file(a.rweights)

    results = []
    t_start = time.time()
    if a.workers > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers, initializer=worker_init,
                     initargs=(a.moddir, a.weights, a.rweights)) as pool:
            for i, rec in enumerate(pool.imap_unordered(run_root, tasks)):
                results.append(rec)
                if (i + 1) % 500 == 0:
                    print(f"  {i + 1}/{len(tasks)} roots, "
                          f"{time.time() - t_start:.0f}s", flush=True)
    else:
        worker_init(a.moddir, a.weights, a.rweights)
        for task in tasks:
            results.append(run_root(task))
    results.sort(key=lambda r: r["root_index"])

    with open(os.path.join(a.out, "results.jsonl"), "w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    def cnt(pred):
        return sum(1 for r in results if pred(r))

    summary = {
        "n_roots": len(results),
        "changed_value_const": cnt(lambda r: r["changed_value_const"]),
        "changed_value_heur": cnt(lambda r: r["changed_value_heur"]),
        "changed_const_heur": cnt(lambda r: r["changed_const_heur"]),
        "gate_pass": cnt(lambda r: r["changed_value_const"]) > 0,
        "searched_per_arm": {arm: cnt(lambda r, arm=arm:
                                      r["arms"][arm]["searched"] > 0)
                             for arm in ARMS},
        "agent_changed_per_arm": {arm: sum(r["arms"][arm]["changed"]
                                           for r in results) for arm in ARMS},
        "root_errors_per_arm": {arm: sum(r["arms"][arm]["root_errors"]
                                         for r in results) for arm in ARMS},
        "errors_per_arm": {arm: sum(r["arms"][arm]["errors"]
                                    for r in results) for arm in ARMS},
        "mean_dt_s_per_arm": {arm: round(
            sum(r["arms"][arm]["dt_s"] for r in results) /
            max(len(results), 1), 4) for arm in ARMS},
        "wall_s": round(time.time() - t_start, 1),
        "workers": a.workers,
        "source": src_desc,
        "md5": file_md5,
    }
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    ch = [r for r in results if r["changed_value_const"]][:12]
    un = [r for r in results if not r["changed_value_const"]][:12]
    with open(os.path.join(a.out, "examples.json"), "w") as f:
        json.dump({"changed_value_const": ch, "unchanged_value_const": un},
                  f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
