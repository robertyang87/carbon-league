"""Short paired Sapling test of one fixed loaded-collector home-entry logit bonus."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

import defense_probe as harness  # fixes numerical-library threads before importing NumPy
import numpy as np

from intent_policy_probe import friendly_conflicts
from opponent_intent import DEST

SEEDS = tuple(range(620099000, 620099004))
HORIZON = 100
BONUS = 1.0
SATURATION_CARGO = 100.0


def choose(features, obs, inputs, policy, enabled):
    me = obs["player"]
    _, bases, workers, _ = obs["players"][me]
    enemy_cells = {w[0] for w in obs["players"][1 - me][2].values()}
    occupied = {w[0] for w in workers.values()} | enemy_cells
    homes = set(bases.values())
    clear = {h for h in homes if h not in occupied and enemy_cells.isdisjoint(DEST[h])}
    feat = sys.modules["cnet2_feat"]
    records = []

    def hook(slot, logits, prefix, baseline):
        if slot >= 10:
            return None
        uid = features.own_uid[slot]
        pos, cargo, kind = workers[uid]
        if kind != "COLLECTOR" or cargo <= 0 or pos in homes:
            return None
        adjacent = homes.intersection(DEST[pos, 1:])
        if len(adjacent) != 1:
            return None
        home = int(next(iter(adjacent)))
        if home not in clear or prefix.reserved[features.cpos(home)]:
            return None
        world = next(a for a in range(1, 5) if DEST[pos, a] == home)
        action = feat.CANON_MOVE[world] if features.seat else world
        if logits[action] < -1.e8:
            return None
        bonus = BONUS * min(cargo / SATURATION_CARGO, 1.)
        adjusted = logits.copy()
        adjusted[action] += bonus
        winner = int(adjusted.argmax())
        records.append({"uid": uid, "home": home, "cargo": cargo, "bonus": bonus,
                        "world_home_move": feat.MOVES[world], "home_action": int(action),
                        "original": int(baseline), "winner": winner})
        return winner if enabled and winner != baseline else None

    labels, _, _ = policy.act(inputs, features.aux, teacher=2, hook=hook)
    joint = features.to_actions(labels)
    flips = [r for r in records if r["winner"] != r["original"]]
    reject = False
    baseline_joint = None
    if enabled and flips:
        original, _, _ = policy.act(inputs, features.aux, teacher=2)
        baseline_joint = features.to_actions(original)
        new_conflicts = friendly_conflicts(obs, joint)
        diverted_homes = {r["home"] for r in flips}
        if new_conflicts - friendly_conflicts(obs, baseline_joint) or any(
                c[0] in diverted_homes for c in new_conflicts):
            joint = baseline_joint
            reject = True
    return joint, baseline_joint, records, reject


def play(seed, side, enabled, pack, policy, sim_type, observer_type):
    random.seed(seed * 7919 + side * 104729 + 31)
    sim = sim_type().reset(seed=seed)
    assert sim.config["size"] == 15
    opponent = harness.load(harness.RULE_OPPONENT, "return_probe_opponent").make_agent()
    observer = observer_type(sim_type)
    features = sys.modules["cnet2_feat"].Feat(side)
    trace = hashlib.sha256()
    tree_income = [0., 0.]
    counts = Counter()
    evidence = []
    for t in range(HORIZON):
        obs = sim.obs(side)
        joint, original, records, reject = choose(features, obs, features.observe(obs), policy, enabled)
        if not enabled and joint != pack.agent(obs, sim.config):
            raise AssertionError("Disabled bonus differs from the shipped Sapling")
        actions = [{}, {}]
        actions[side] = joint
        actions[1 - side] = opponent(sim.obs(1 - side), sim.config)
        trace.update((json.dumps(actions, sort_keys=True) + "\n").encode())
        before = len(observer.events)
        observer.step(sim, actions)
        deposits = {e["worker"]: e["amount"] for e in observer.events[before:]
                    if e["event"] == "deposit" and e["player"] == side}
        for tid, tree in sim.trees.items():
            tree_income[tree.player] += sim.trees_dict[tid][1]
        counts["joint_conflict_rejections"] += reject
        counts["changed_joint_actions"] += original is not None and joint != original
        for r in records:
            changed = r["original"] != r["winner"]
            sent = joint.get(r["uid"]) == r["world_home_move"]
            reference_sent = (original.get(r["uid"], "NONE") == r["world_home_move"]
                              if original is not None else sent)
            executed_flip = bool(changed and enabled and not reject and sent and not reference_sent)
            counts["eligible_slots"] += 1
            counts["local_preference_flips"] += changed
            counts["home_commands"] += sent
            counts["applied_home_flips"] += executed_flip
            amount = deposits.get(r["uid"], 0.) if sent else 0.
            counts["same_step_normal_deposits_from_eligible"] += amount > 0
            counts["same_step_normal_deposits_from_flips"] += executed_flip and amount > 0
            if executed_flip:
                counts["same_step_deposit_carbon_from_flips"] += amount
            evidence.append({**r, "t": t, "rejected": reject, "submitted_home": sent,
                             "executed_flip": executed_flip, "reference_sent_home": reference_sent,
                             "deposited": amount})
        if any(not sim.player_workers[p] and not sim.player_trees[p] and
               (not sim.bases[p] or sim.cash[p] < min(sim.config["recCollectorCost"],
                                                      sim.config["recPlanterCost"])) for p in (0, 1)):
            break
    economics = {}
    for label, p in (("own", side), ("opponent", 1 - side)):
        economics[label] = {"delivery": observer.d[p]["own_positive_delivery_carbon"],
                            "cash": sim.cash[p], "tree_absorption_income": tree_income[p],
                            "completed_loop": bool(observer.d[p]["closed_loop_events"])}
    return {"steps": sim.step_count, "counts": dict(counts), "economics": economics,
            "action_sha256": trace.hexdigest(), "opportunities": evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pack = harness.load(harness.PACK, "return_sapling")
    if pack._OPTS.get("teacher") != 2 or any(v for k, v in pack._OPTS.items() if k != "teacher"):
        raise AssertionError("Unexpected shipped options")
    sim_path = harness.ROOT / "code/fastsim.py"
    obs_path = harness.ROOT / "reports/sapling_bootstrap/smoke_ab.py"
    sim = harness.load(sim_path, "return_simulator")
    observer = harness.load(obs_path, "return_observer")
    weights = pack._dequantize(pack.zlib.decompress(pack.base64.b64decode(pack._W64)), pack._QMETA)
    weights["meta"] = np.array(pack._WMETA)
    policy = sys.modules["cnet2_np"].Policy(weights)
    rows = []
    for seed in SEEDS:
        for seat in (0, 1):
            pair = {"seed": seed, "seat": seat}
            for arm, enabled in (("original", False), ("bonus", True)):
                pair[arm] = play(seed, seat, enabled, pack, policy, sim.FastCarbon, observer.Observer)
            if not pair["bonus"]["counts"].get("changed_joint_actions", 0):
                if pair["original"]["action_sha256"] != pair["bonus"]["action_sha256"]:
                    raise AssertionError("Paired trajectories diverged without an intervention")
            rows.append(pair)
    summaries = {}
    for arm in ("original", "bonus"):
        totals = Counter()
        for pair in rows:
            totals.update(pair[arm]["counts"])
        complete = [p[arm] for p in rows if p[arm]["steps"] == HORIZON]
        econ = {}
        for label in ("own", "opponent"):
            v = [r["economics"][label] for r in complete]
            econ[label] = {"delivery_sum": sum(x["delivery"] for x in v),
                           "tree_absorption_sum": sum(x["tree_absorption_income"] for x in v),
                           "cash_mean": sum(x["cash"] for x in v) / len(v) if v else None,
                           "games_with_loop": sum(x["completed_loop"] for x in v)}
        summaries[arm] = {"counter_totals": dict(totals), "t100_games": len(complete),
                          "economics_at_t100": econ}
    result = {
        "seeds": SEEDS, "turns": HORIZON, "bonus_cap": BONUS, "cargo_scale": SATURATION_CARGO,
        "scope": "Short paired exploratory games, four maps x both seats, responsive rule opponent. "
                 "Immediate deposits are actual transfers, not guaranteed incremental profit. "
                 "Post-intervention eligibility sets can differ. Per-game observation stops at extinction.",
        "sources": {p.name: harness.digest(p) for p in (harness.PACK, harness.RULE_OPPONENT, sim_path, obs_path)},
        "summary": summaries, "pairs": rows,
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
