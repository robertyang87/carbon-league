"""Two independent opening-economy ablations on the released 1.2M Sapling.

Run from any directory: python harvest_probe.py --out /tmp/harvest_probe.json.
All policy decisions use the active player's observation. Exact simulator events
are read only after a turn, for accounting. No parameter fit or exam seeds.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
TACTICS = HERE.parent
sys.path.insert(0, str(TACTICS))
import defense_probe as harness
import numpy as np
from defense import DefenseBoard
from intent_policy_probe import friendly_conflicts
from opponent_intent import DEST, DIST, MOVES

SEEDS = tuple(range(620101000, 620101004))
HORIZON = 100
ARMS = ("original", "safe_harvest", "working_capital_return")


def candidate(features, policy, obs, config, arm, original):
    """Apply a bounded bonus before downstream autoregressive slots commit."""
    me = obs["player"]
    cash, bases, workers, _ = obs["players"][me]
    homes = list(bases.values())
    nearest = DIST[:, homes].min(axis=1)
    # Shipped game: flat enrollment cost. Refuse a different fee regime.
    if config["recCollectorIncreaseCost"] or config["recPlanterIncreaseCost"]:
        raise ValueError("Expected flat recruitment costs")
    fee = min(config["recCollectorCost"], config["recPlanterCost"])
    blocked_harvest = set()
    for _, player_bases, _, trees in obs["players"]:
        blocked_harvest.update(player_bases.values())
        blocked_harvest.update(tree[0] for tree in trees.values())
    danger = DefenseBoard(obs, config)
    geom = sys.modules["cnet2_feat"]
    observations = []
    inputs = features.observe(obs)

    def hook(slot, logits, prefix, planned):
        if slot >= 10:
            return None
        uid = features.own_uid[slot]
        pos, cargo, kind = workers[uid]
        if kind != "COLLECTOR":
            return None
        visible_carbon = obs["carbon"][pos]
        start = (obs["step"] < 30 and cargo == 0 and visible_carbon >= 30
                 and pos not in blocked_harvest)
        back = (cash < fee <= cargo and 2 <= nearest[pos] <= 5)
        if not start and not back:
            return None
        opportunities = {}
        for rule, applies in (("safe_harvest", start), ("working_capital_return", back)):
            if not applies:
                continue
            allowed = []
            for action in range(5):
                world = geom.CANON_MOVE[action] if features.seat else action
                target = int(DEST[pos, world])
                if rule == "safe_harvest" and world != 0:
                    continue
                if rule == "working_capital_return" and nearest[target] != nearest[pos] - 1:
                    continue
                risk = danger.assess(uid, MOVES[world])
                if (logits[action] < -1e8 or prefix.reserved[features.cpos(target)]
                        or risk.collision_loss_possible or risk.opponent_base or risk.enemy_tree_count):
                    continue
                allowed.append(action)
            if allowed:
                # Two separate interventions; a fixed, modest 1-logit cap.
                bonus = min(visible_carbon/60, 1.0) if rule == "safe_harvest" else min(cargo/(2*fee), 1.0)
                adjusted = logits.copy()
                adjusted[allowed] += bonus
                winner = int(adjusted.argmax())
                opportunities[rule] = winner
                observations.append({
                    "rule": rule, "uid": uid, "cell": int(pos), "pre_cash": float(cash),
                    "pre_cargo": float(cargo), "visible_carbon": float(visible_carbon),
                    "pre_distance": int(nearest[pos]), "bonus": float(bonus),
                    "baseline": int(planned), "preferred": winner,
                    "safe_candidates": [int(a) for a in allowed],
                })
        winner = opportunities.get(arm, planned)
        return winner if winner != planned else None

    labels, _, _ = policy.act(inputs, features.aux, teacher=2, hook=hook)
    proposed = features.to_actions(labels)
    if arm == "original" and proposed != original:
        raise AssertionError("Decoded shadow no longer matches shipped int8 pack")
    conflicts = friendly_conflicts(obs, proposed)
    flipped = [row for row in observations if row["rule"] == arm and row["baseline"] != row["preferred"]]
    target_cells = {int(DEST[workers[r["uid"]][0], MOVES.index(proposed.get(r["uid"], "NONE"))]) for r in flipped}
    reject = bool(flipped and (conflicts - friendly_conflicts(obs, original)
                               or any(c[0] in target_cells for c in conflicts)))
    return original if reject else proposed, observations, reject


def play(seed, seat, arm, pack, policy, simulator, smoke):
    random.seed(seed * 7919 + seat * 104729 + 31)
    sim = simulator.FastCarbon().reset(seed=seed)
    foe = harness.load(harness.RULE_OPPONENT, "harvest_opponent").make_agent()
    observer = smoke.Observer(simulator.FastCarbon)
    features = sys.modules["cnet2_feat"].Feat(seat)
    events, changes, first = [], [], {}
    counts = Counter()
    joint_trace = hashlib.sha256()
    for t in range(HORIZON):
        obs = sim.obs(seat)
        _, bases, workers, _ = obs["players"][seat]
        original = pack.agent(obs, sim.config)
        chosen, records, rejected = candidate(features, policy, obs, sim.config, arm, original)
        actions = [{}, {}]
        actions[seat] = chosen
        actions[1-seat] = foe(sim.obs(1-seat), sim.config)
        joint_trace.update(json.dumps(actions, sort_keys=True).encode() + b"\n")
        pre_workers = {uid for uid, unit in workers.items() if unit[2] == "COLLECTOR"}
        for uid in sorted(pre_workers):
            pos, cargo, _ = workers[uid]
            counts["collector_turns"] += 1
            if obs["carbon"][pos] > 0:
                counts["collector_on_positive_carbon"] += 1
            if cargo > 0:
                counts["loaded_turns"] += 1
                dist = min(DIST[pos, h] for h in bases.values())
                if dist == 1:
                    counts["loaded_adjacent_own_base"] += 1
                    first.setdefault("loaded_adjacent_state", t)
        before = len(observer.events)
        observer.step(sim, actions)
        after = [e for e in observer.events[before:] if e["player"] == seat]
        events.extend(after)
        for event in after:
            first.setdefault(event["event"] + "_resolution", t + 1)
        counts["collector_deaths_all_causes"] += len(pre_workers - set(sim.player_workers[seat]))
        counts["joint_rejections"] += rejected
        counts["changed_joint_turns"] += chosen != original
        for row in records:
            rule = row["rule"]
            counts[rule + "_opportunities"] += 1
            uid = row["uid"]
            if arm == rule and row["baseline"] != row["preferred"] and not rejected:
                if original.get(uid, "NONE") != chosen.get(uid, "NONE"):
                    changes.append(dict(row, t=t, submitted=chosen.get(uid, "NONE"),
                                        original_action=original.get(uid, "NONE"),
                                        alive_after=uid in sim.player_workers[seat],
                                        collected=sum(e["amount"] for e in after if e["event"] == "collect" and e["worker"] == uid)))
        if any(not sim.player_workers[p] and not sim.player_trees[p]
               and (not sim.bases[p] or sim.cash[p] < min(sim.config["recCollectorCost"], sim.config["recPlanterCost"]))
               for p in (0, 1)):
            break
    # Later bank receipts are descriptive; overlapping interventions may lead to the same deposit.
    for change in changes:
        dep = next((e for e in events if e["event"] == "deposit" and e["worker"] == change["uid"]
                    and e["step"] >= change["t"] + 1), None)
        change["next_normal_deposit_before_stop"] = dep
    final = {}
    for name, p in (("own", seat), ("opponent", 1-seat)):
        ledger = observer.d[p]
        final[name] = {"cash": sim.cash[p], "total_harvest": ledger["collected_carbon"],
                       "normal_deposits": ledger["positive_delivery_events"],
                       "delivered": ledger["own_positive_delivery_carbon"],
                       "completed_loop": bool(ledger["closed_loop_events"]),
                       "remaining_cargo": sum(sim.workers[w].carbon for w in sim.player_workers[p])}
    return dict(steps=sim.step_count, first=first, counts=dict(counts), final=final, changes=changes,
                joint_sha256=joint_trace.hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    pack = harness.load(harness.PACK, "detail_harvest_sapling")
    if pack._OPTS["teacher"] != 2 or any(v for k, v in pack._OPTS.items() if k != "teacher"):
        raise ValueError("Unexpected Sapling pack")
    sim_file = harness.ROOT / "code/fastsim.py"
    obs_file = harness.ROOT / "reports/sapling_bootstrap/smoke_ab.py"
    simulator = harness.load(sim_file, "detail_harvest_fastsim")
    smoke = harness.load(obs_file, "detail_harvest_observer")
    weights = pack._dequantize(pack.zlib.decompress(pack.base64.b64decode(pack._W64)), pack._QMETA)
    weights["meta"] = np.array(pack._WMETA)
    policy = sys.modules["cnet2_np"].Policy(weights)
    pairs = []
    for seed in SEEDS:
        for seat in (0, 1):
            pair = {arm: play(seed, seat, arm, pack, policy, simulator, smoke) for arm in ARMS}
            for arm in ARMS[1:]:
                if not pair[arm]["counts"]["changed_joint_turns"]:
                    assert pair[arm]["joint_sha256"] == pair["original"]["joint_sha256"]
            pairs.append(dict(seed=seed, seat=seat, **pair))
    summary = {}
    for arm in ARMS:
        totals = Counter()
        for pair in pairs:
            row = pair[arm]
            totals.update(row["counts"])
            totals.update(steps=row["steps"], games_with_loop=row["final"]["own"]["completed_loop"],
                          activated_games=bool(row["changes"]), direct_flips=len(row["changes"]))
        summary[arm] = dict(totals=totals,
                            economics={p: {k:sum(pair[arm]["final"][p][k] for pair in pairs)
                                           for k in pairs[0][arm]["final"][p]} for p in ("own","opponent")})
    evidence = TACTICS / "opening_probe.json"
    zero = [r["original"] for r in json.loads(evidence.read_text())["pairs"]
            if r["original"]["snapshots"]["60"]["own_positive_delivery_carbon"] == 0]
    result = dict(seeds=SEEDS, horizon=HORIZON, source_sha256={f.name:harness.digest(f) for f in
                  (harness.PACK, harness.RULE_OPPONENT, sim_file, obs_file, evidence, Path(__file__),
                   TACTICS/"intent_policy_probe.py", TACTICS/"defense.py")},
                  diagnosis_previous_zero_delivery_t60=[dict(seed=a["seed"], seat=a["seat"],
                  harvest=a["snapshots"]["60"]["collected_carbon"], carried=a["snapshots"]["60"]["cargo"]) for a in zero],
                  protocol={"safe_harvest":"t<30; empty collector on visible ground >=30, no base/tree. Favor staying by min(carbon/60,1).",
                            "working_capital_return":"cash below one default hire, cargo covers at least one hire, distance 2..5 to nearest own base. Favor all safe moves decreasing distance by one by min(cargo/(2*hire),1).",
                            "shared":"Reject reserved, enemy-base, vulnerable-to-collision and enemy-tree-exposed target; reject joint conflicts introduced by a change. Two separate arms. Shipped pack verified every baseline call; responsive opponent on separate maps only. All arms end at t100 or extinction. Counterfactual post-intervention trajectories diverge; opportunities are arm-specific. Summary economy columns are sums per arm, cash at actual stop, action counters are not independent trials. No claims about wins."},
                  summary=summary,pairs=pairs)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
