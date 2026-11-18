"""Paired short-game test of a one-ply cargo defense rule on Sapling's shipped int8 weights.

Run: python reports/master_family/tactical_rules/defense_probe.py --out /tmp/defense.json
The candidate and opponent see their own observations. The observer alone reads simulator events.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from collections import Counter

for env_key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[env_key] = "1"

import numpy as np
from defense import DefenseBoard

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACK = ROOT / "data/weights/sapling_1200k/agent_int8.py"
RULE_OPPONENT = ROOT / "code/policy_v0_5.py"
SEEDS = tuple(range(620092000, 620092004))
HORIZON = 100


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def defense_hook(features, obs, config, enabled, tally):
    """A worker change is seen by all subsequently decoded units and bases."""
    board = DefenseBoard(obs, config)
    feat_module = sys.modules["cnet2_feat"]
    workers = obs["players"][obs["player"]][2]

    def choose(slot, logits, prefix, network_action):
        if slot >= 10:
            return None
        uid = features.own_uid[slot]
        _, cargo, kind = workers[uid]
        if kind != "COLLECTOR" or cargo <= 0:
            return None
        tally["loaded_collector_decisions"] += 1

        def score(action):
            world = feat_module.CANON_MOVE[action] if features.seat else action
            return board.assess(uid, feat_module.MOVES[world])

        planned = score(network_action)
        if not planned.collision_loss_possible:
            return None
        tally["possible_loss_on_planned_landing"] += 1
        for action in np.argsort(-logits, kind="stable"):
            if logits[action] < -1e8:
                continue
            risk = score(int(action))
            canon_dest = features.cpos(risk.destination)
            if (risk.collision_loss_possible or risk.opponent_base
                    or prefix.reserved[canon_dest] > 0):
                continue
            tally["has_safer_unreserved_choice"] += 1
            if enabled:
                tally["changed_collector_actions"] += 1
                return int(action)
            break
        return None

    return choose


def game(seed, side, enabled, pack, policy, smoke, sim_type):
    random.seed(seed * 7919 + side * 104729 + 31)
    sim = sim_type().reset(seed=seed)
    features = sys.modules["cnet2_feat"].Feat(side)
    observer = smoke.Observer(sim_type)
    opponent = load(RULE_OPPONENT, "defense_probe_opponent").make_agent()
    tally = Counter()
    t = 0
    extinct = False
    while t < HORIZON and not extinct:
        obs = sim.obs(side)
        inputs = features.observe(obs)
        hook = defense_hook(features, obs, sim.config, enabled, tally)
        labels, _, _ = policy.act(inputs, features.aux, teacher=2, hook=hook)
        candidate_action = features.to_actions(labels)
        if not enabled:
            # The disabled hook must be inert at the actual public entry point.
            if candidate_action != pack.agent(obs, sim.config):
                raise AssertionError("Shadow arm differs from public int8 agent")
            tally["pack_action_parity_calls"] += 1
        actions = [None, None]
        actions[side] = candidate_action
        actions[1-side] = opponent(sim.obs(1-side), sim.config)
        old_owners = {tid: tree.player for tid, tree in sim.trees.items()}
        observer.step(sim, actions)
        for uid, loser, carbon, receiver, winner in sim.last_collisions:
            if loser == side and winner == 1-side:
                tally["workers_killed_by_enemy_collector"] += 1
            if loser == 1-side and winner == side:
                tally["enemy_workers_killed_by_own_collector"] += 1
            if receiver == 1-loser:
                key = "own_collision_cargo_to_enemy" if loser == side else "enemy_collision_cargo_to_own"
                tally[key] += carbon
        for tid, tree in sim.trees.items():
            if tid in old_owners and tree.player != old_owners[tid]:
                key = "own_trees_seized_by_enemy" if tree.player != side else "enemy_trees_seized_by_own"
                tally[key] += 1
        t += 1
        extinct = any(not sim.player_workers[p] and not sim.player_trees[p]
                      and (not sim.bases[p] or sim.cash[p] < min(sim.config["recCollectorCost"],
                                                               sim.config["recPlanterCost"]))
                      for p in (0, 1))
    counts = ("loaded_collector_decisions", "possible_loss_on_planned_landing",
              "has_safer_unreserved_choice", "changed_collector_actions", "pack_action_parity_calls",
              "workers_killed_by_enemy_collector", "enemy_workers_killed_by_own_collector",
              "own_collision_cargo_to_enemy", "enemy_collision_cargo_to_own",
              "own_trees_seized_by_enemy", "enemy_trees_seized_by_own")
    result = {k: tally[k] for k in counts}
    result.update(terminal=extinct, step=t, own_cash=sim.cash[side], enemy_cash=sim.cash[1-side],
                  own_delivery=observer.d[side]["own_positive_delivery_carbon"],
                  enemy_delivery=observer.d[1-side]["own_positive_delivery_carbon"],
                  own_completed_harvest_delivery=bool(observer.d[side]["closed_loop_events"]))
    return result


def attack_fixtures(pack, policy, simulator):
    """All 12 unique distance-1/2 attackers around a loaded collector; all five responses."""
    feat_module = sys.modules["cnet2_feat"]
    rows = []
    for side in (0, 1):
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if not 1 <= abs(dx) + abs(dy) <= 2:
                    continue
                sim = simulator.FastCarbon().reset(full_carbon=[0.] * 225, cash=0.)
                base = sim.bases[side][0][1]
                row, col = divmod(base, 15)
                col = (col + (3 if side == 0 else -3)) % 15
                own_pos = row * 15 + col
                enemy_pos = ((row + dy) % 15) * 15 + ((col + dx) % 15)
                own_id = sim._new_worker_id(side)
                enemy_id = sim._new_worker_id(1-side)
                for uid, player, pos, cargo in ((own_id, side, own_pos, 50.),
                                                (enemy_id, 1-side, enemy_pos, 0.)):
                    sim.workers[uid] = simulator._Worker(uid, player, pos, cargo, True)
                    sim.player_workers[player].append(uid)
                original_state = sim.get_state()
                obs = sim.obs(side)
                record = {"seat": side, "dx_world": dx, "dy_world_row": dy, "by_arm": {}}
                for name in ("original", "defense"):
                    features = feat_module.Feat(side)
                    inputs = features.observe(obs)
                    hook = defense_hook(features, obs, sim.config, name == "defense", Counter())
                    labels, _, _ = policy.act(inputs, features.aux, teacher=2, hook=hook)
                    act = features.to_actions(labels)
                    if name == "original" and act != pack.agent(obs, sim.config):
                        raise AssertionError("Fixture baseline differs from public pack")
                    fatal_responses = []
                    for enemy_move in feat_module.MOVES:
                        trial = simulator.FastCarbon().set_state(original_state)
                        actions = [{}, {}]
                        actions[side] = act
                        actions[1-side] = {} if enemy_move == "NONE" else {enemy_id: enemy_move}
                        trial.step(*actions)
                        if own_id not in trial.workers:
                            fatal_responses.append(enemy_move)
                    record["by_arm"][name] = {
                        "collector_action": act.get(own_id, "NONE"),
                        "killing_opponent_responses": fatal_responses,
                    }
                rows.append(record)
    return {
        "scope": "Constructed micropositions, not sampled from play; seat-relative home offset, one 50-carbon "
                 "collector and one empty enemy collector, empty board. Exhaust all distance-one/two attacker cells "
                 "and all five simultaneous enemy actions. The policy sees position only. "
                 "Branches are possible responses, not independent games or empirical action probabilities.",
        "positions": len(rows),
        "enemy_response_branches_per_arm": len(rows) * 5,
        "changed_positions": sum(r["by_arm"]["original"]["collector_action"] !=
                                 r["by_arm"]["defense"]["collector_action"] for r in rows),
        "positions_with_any_fatal_response": {
            arm: sum(bool(r["by_arm"][arm]["killing_opponent_responses"]) for r in rows)
            for arm in ("original", "defense")
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new output path")
    start = time.monotonic()
    pack = load(PACK, "defense_probe_sapling")
    if pack._OPTS.get("teacher") != 2 or any(v for k, v in pack._OPTS.items() if k != "teacher"):
        raise AssertionError("Unexpected pack options")
    simulator = load(ROOT / "code/fastsim.py", "defense_probe_fastsim")
    sim_type = simulator.FastCarbon
    smoke = load(ROOT / "reports/sapling_bootstrap/smoke_ab.py", "defense_probe_observer")
    # Source and weights are those embedded in the public single-file package.
    weights = pack._dequantize(pack.zlib.decompress(pack.base64.b64decode(pack._W64)), pack._QMETA)
    weights["meta"] = np.array(pack._WMETA)
    policy = sys.modules["cnet2_np"].Policy(weights)
    rows = []
    for seed in SEEDS:
        for seat in (0, 1):
            entry = {"seed": seed, "seat": seat}
            for arm in ("original", "defense"):
                entry[arm] = game(seed, seat, arm == "defense", pack, policy, smoke, sim_type)
            rows.append(entry)
    totals = {arm: {key: sum(row[arm][key] for row in rows) for key in rows[0][arm]}
              for arm in ("original", "defense")}
    output = {"scope": "Exploratory held-out-by-map, 100 real simultaneous transitions maximum. "
                       "Four independent maps; two seats per map. Opponent responds on each arm's own trajectory. "
                       "No top2 training or examiner seeds. Terminal snapshots stop at actual elimination. "
                       "Cash at stop is not official full-game score. Counters are over each arm's own exposure.",
              "intervention": "Loaded collectors whose chosen landing can lose in a head-to-head one-step response. "
                              "Select highest-policy-logit safe unreserved landing outside enemy bases, otherwise retain. "
                              "Reservation covers earlier own units; later units and bases re-decode with updated prefix. "
                              "The model itself can still collide with a teammate. No opponent action supplied.",
              "sources_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                                 (HERE / "defense.py", Path(__file__).resolve(), PACK, RULE_OPPONENT,
                                  ROOT / "code/fastsim.py", ROOT / "reports/sapling_bootstrap/smoke_ab.py")},
              "constructed_attacks": attack_fixtures(pack, policy, simulator),
              "seeds": SEEDS, "horizon": HORIZON, "rows": rows, "totals": totals,
              "wall_seconds": round(time.monotonic()-start, 3)}
    with args.out.open("x") as file:
        json.dump(output, file, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
