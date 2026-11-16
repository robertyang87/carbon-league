#!/usr/bin/env python3
"""Part 2 (PREREG_SEARCH §2): root-level quality check by REAL
continuation of the roots the fitted-scale value leaf rewrites.

For every root where the value arm (M_eq = a*z, fitted scale) and the
heuristic-leaf default arm chose different actions in the gate-caliber
ablation (out_gate_a/results.jsonl, changed_value_heur), play the game
out twice from the recorded root state:

  branch "new"     : root seat's root-step action = value arm's action
  branch "default" : root seat's root-step action = heur arm's action

Everything else is ONE instrument used symmetrically in both branches:
both seats are played by the Echo3 deployment pack itself
(production CONF, SEARCH=False) -- including the opponent's root-step
action (computed inside each branch from the pristine root state;
the two branches' opponent root actions are asserted equal). After
the forced root step, both seats are the pack's until the game ends.

State: FastCarbon.load_observation on the replay's recorded full
observation (seat-0 entry, carries full_carbon); the search layer's
+500 tid-counter bump is mirrored. Terminal scoring:
  - extinction (either side, _extinct as in cnet2_search): F01 rule
    from leaf() -- extinct side scores (step_count - episodeSteps - 1),
    survivor keeps the cash it carried INTO the terminal step;
  - step limit (step_count reaches episodeSteps): scores are the
    cashes at that state (verified against recorded games: final
    recorded cash == meta scores on step-limit games).
U from the root seat's margin by the platform formula
(cnet2_build._u_of_margin). dU = U_new - U_default per root.
"""
import argparse
import copy
import gzip
import importlib.util
import json
import math
import os
import sys

SEED_BASE = 20261115
EPISODE_STEPS = 300

G = {}


def u_of_margin(m):
    w = 1.0 if m > 0 else (0.0 if m < 0 else 0.5)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-float(m) / 1000.0))


def extinct(sim, p):
    """cnet2_search._extinct, verbatim logic."""
    if any(w.player == p for w in sim.workers.values()):
        return False
    if any(t.player == p for t in sim.trees.values()):
        return False
    if not sim.bases[p]:
        return True
    cheapest = min(sim.config.get("recCollectorCost", 30),
                   sim.config.get("recPlanterCost", 30))
    return float(sim.cash[p]) < cheapest


def load_pack(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def worker_init(moddir, pack_path, replays_dir):
    sys.path.insert(0, moddir)
    import numpy as np
    from fastsim import FastCarbon
    G["np"] = np
    G["FastCarbon"] = FastCarbon
    G["pack"] = {0: load_pack("echo3_seat0", pack_path),
                 1: load_pack("echo3_seat1", pack_path)}
    G["replays"] = replays_dir


def play_out(root, forced_action):
    """One continuation. Returns (margin_from_root_seat, end_kind,
    end_step, opp_root_action_canon)."""
    np = G["np"]
    seat = root["seat"]
    opp = 1 - seat
    sim = G["FastCarbon"]()
    sim.load_observation(root["full_obs"])
    sim._tid_counter = [c + 500 for c in sim._tid_counter]
    for p in (0, 1):
        G["pack"][p]._AG = None     # fresh instrument state per game
    np.random.seed(SEED_BASE + root["root_index"])

    def terminal_margin(prev_cash):
        if extinct(sim, 0) or extinct(sim, 1):
            pen = float(sim.step_count) - float(EPISODE_STEPS) - 1.0
            rew = [pen if extinct(sim, p) else float(prev_cash[p])
                   for p in (0, 1)]
            return rew[seat] - rew[opp], "extinct"
        if sim.step_count >= EPISODE_STEPS:
            return float(sim.cash[seat]) - float(sim.cash[opp]), "limit"
        return None

    # root step: opponent action from the pack on the pristine state
    opp_act = G["pack"][opp].agent(sim.obs(opp), None)
    acts = [None, None]
    acts[seat] = forced_action
    acts[opp] = opp_act
    prev_cash = list(sim.cash)
    sim.step(acts[0], acts[1])
    r = terminal_margin(prev_cash)
    steps = 0
    while r is None:
        prev_cash = list(sim.cash)
        a0 = G["pack"][0].agent(sim.obs(0), None)
        a1 = G["pack"][1].agent(sim.obs(1), None)
        sim.step(a0, a1)
        r = terminal_margin(prev_cash)
        steps += 1
        if steps > EPISODE_STEPS + 2:
            raise RuntimeError("continuation overrun")
    margin, kind = r
    return margin, kind, int(sim.step_count), \
        json.dumps(opp_act or {}, sort_keys=True)


def run_root(root):
    np = G["np"]
    path = os.path.join(G["replays"], root["game_key"])
    d = json.load(gzip.open(path, "rt"))
    full = d["steps"][root["step"]][0]["observation"]
    root = dict(root, full_obs=full)
    out = {"game_key": root["game_key"], "seat": root["seat"],
           "step": root["step"], "root_index": root["root_index"]}
    res = {}
    for branch, act in (("new", root["act_value"]),
                        ("default", root["act_heur"])):
        margin, kind, end_step, opp_canon = play_out(root, act)
        res[branch] = {"margin": margin, "U": u_of_margin(margin),
                       "end": kind, "end_step": end_step,
                       "opp_root_action": opp_canon}
    out["new"] = res["new"]
    out["default"] = res["default"]
    out["opp_root_action_symmetric"] = \
        res["new"]["opp_root_action"] == res["default"]["opp_root_action"]
    out["dU"] = res["new"]["U"] - res["default"]["U"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moddir", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--replays", required=True)
    ap.add_argument("--gate-results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    roots = []
    for line in open(a.gate_results):
        r = json.loads(line)
        if r["changed_value_heur"]:
            roots.append({
                "game_key": r["game_key"], "seat": r["seat"],
                "step": r["step"], "root_index": r["root_index"],
                "act_value": json.loads(r["arms"]["value"]["action"]),
                "act_heur": json.loads(r["arms"]["heur"]["action"]),
            })
    print(f"changed roots (value vs heur): {len(roots)}", flush=True)

    results = []
    if a.workers > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers, initializer=worker_init,
                     initargs=(a.moddir, a.pack, a.replays)) as pool:
            for i, rec in enumerate(pool.imap_unordered(run_root, roots)):
                results.append(rec)
                if (i + 1) % 25 == 0:
                    print(f"  {i + 1}/{len(roots)}", flush=True)
    else:
        worker_init(a.moddir, a.pack, a.replays)
        for root in roots:
            results.append(run_root(root))
    results.sort(key=lambda r: r["root_index"])

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    import numpy as np
    du = np.array([r["dU"] for r in results])
    summary = {
        "n_changed_roots": int(len(results)),
        "dU_mean": float(du.mean()),
        "dU_q25": float(np.percentile(du, 25)),
        "dU_median": float(np.percentile(du, 50)),
        "dU_q75": float(np.percentile(du, 75)),
        "neg_fraction": float((du < 0).mean()),
        "zero_fraction": float((du == 0).mean()),
        "pos_fraction": float((du > 0).mean()),
        "opp_root_action_symmetric_all":
            bool(all(r["opp_root_action_symmetric"] for r in results)),
        "end_kinds": {k: int(sum(1 for r in results
                                 if r["new"]["end"] == k))
                      for k in ("extinct", "limit")},
    }
    print(json.dumps(summary, indent=1), flush=True)
    with open(os.path.join(os.path.dirname(a.out),
                           "continuation_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
