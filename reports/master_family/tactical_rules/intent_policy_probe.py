"""Test a frozen probability-weighted tree-defense logit in the shipped Sapling decoder."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

import defense_probe as harness  # fixes BLAS threads before importing NumPy
import numpy as np

from defense import DefenseBoard
from opponent_intent import DEST, DIST, MOVES, MOVE_ID, predict
from tree_screen_probe import CASES, root_position

HERE = Path(__file__).resolve().parent
ROOT = harness.ROOT
SEEDS = tuple(range(620096000, 620096004))
BONUS = 1.5
AGENT = ROOT / "code/policy_raid_v0_1.py"


def friendly_conflicts(obs, action):
    """Potential co-arrivals involving existing workers or ordered recruits."""
    _, bases, workers, _ = obs["players"][obs["player"]]
    buckets = {}
    for uid, (pos, _, _) in workers.items():
        dest = int(DEST[pos, MOVE_ID[action.get(uid, "NONE")]])
        buckets.setdefault(dest, []).append(uid)
    for bid, pos in bases.items():
        if action.get(bid) in ("RECCOLLECTOR", "RECPLANTER"):
            buckets.setdefault(pos, []).append(bid)
    return {(pos, *sorted((a, b))) for pos, members in buckets.items()
            for i, a in enumerate(members) for b in members[i + 1:]}


def choose(features, inputs, obs, previous, config, policy, weights, enabled, *, intent_filter=None):
    side = obs["player"]
    _, bases, workers, trees = obs["players"][side]
    enemy_cash, _, foes, _ = obs["players"][1 - side]
    lifespan = config["treeLifespan"]
    ages = {pos: age for pos, age in trees.values() if age < lifespan}
    targets = {pos: uid for uid, (pos, _, kind) in foes.items() if kind == "PLANTER" and pos in ages}
    eligible = targets and enemy_cash >= config["seizeCost"]
    board = DefenseBoard(obs, config) if eligible else None
    probabilities = predict(obs, weights, previous) if eligible else {}
    if eligible and intent_filter is not None:
        probabilities = intent_filter(obs, config, probabilities)
    moves = sys.modules["cnet2_feat"]
    logs = []

    def apply(slot, logits, prefix, net_action):
        if not eligible or slot >= 10:
            return None
        uid = features.own_uid[slot]
        pos, cargo, role = workers[uid]
        if role != "COLLECTOR":
            return None
        adjusted = logits.copy()
        candidates = []
        for a in range(5):
            if logits[a] < -1e8:
                continue
            world_a = int(moves.CANON_MOVE[a]) if side else a
            target = int(DEST[pos, world_a])
            if target not in targets:
                continue
            foe = targets[target]
            risk = board.assess(uid, MOVES[world_a])
            if not risk.can_screen_tree or risk.opponent_base or prefix.reserved[features.cpos(target)]:
                continue
            stay_p = float(probabilities[foe][0])
            remaining = min(lifespan - ages[target], config["episodeSteps"] - 1 - obs["step"])
            # Banking immediately is valuable; a loaded collector next to home receives less incentive.
            can_bank = cargo > 0 and any(DIST[pos, home] <= 1 for home in bases.values())
            discount = 1 / (1 + cargo / 30) if can_bank else 1.0
            extra = BONUS * stay_p * max(0, remaining) / lifespan * discount
            if extra <= 0:
                continue
            adjusted[a] += extra
            candidates.append({"canonical_action": a, "world_action": MOVES[world_a], "target": target,
                               "intruder": foe, "stay_p": stay_p, "bonus": extra,
                               "logit_before": float(logits[a]), "logit_after": float(adjusted[a])})
        if candidates:
            winner = int(adjusted.argmax())
            logs.append({"collector": uid, "slot": slot, "network_canonical": net_action,
                         "winner_canonical": winner, "candidates": candidates})
            if enabled and winner != net_action:
                return winner
        return None

    y, _, _ = policy.act(inputs, features.aux, teacher=2, hook=apply)
    chosen = features.to_actions(y)
    changed_proposals = [r for r in logs if r["winner_canonical"] != r["network_canonical"]]
    reject = False
    if enabled and changed_proposals:
        original_y, _, _ = policy.act(inputs, features.aux, teacher=2)
        original = features.to_actions(original_y)
        conflicts = friendly_conflicts(obs, chosen)
        protected = {c["target"] for r in changed_proposals for c in r["candidates"]
                     if c["canonical_action"] == r["winner_canonical"]}
        if conflicts - friendly_conflicts(obs, original) or any(pos in protected for pos, _, _ in conflicts):
            chosen = original
            reject = True
    return chosen, logs, reject


def record(logs, reject, enabled, obs, actions, sim, counts):
    side = obs["player"]
    for r in logs:
        own_action = actions[side].get(r["collector"], "NONE")
        counts["adjusted_logits"] += len(r["candidates"])
        counts["affected_collector_slots"] += 1
        flip = r["network_canonical"] != r["winner_canonical"]
        counts["local_argmax_flips"] += int(flip)
        counts["applied_argmax_flips"] += int(flip and enabled and not reject)
        for c in r["candidates"]:
            stayed = actions[1-side].get(c["intruder"], "NONE") == "NONE"
            screened = own_action == c["world_action"]
            tree_id, age_before = next((tid, age) for tid, (pos, age)
                                       in obs["players"][side][3].items() if pos == c["target"])
            tree_after = sim.trees.get(tree_id)
            seized = tree_after is not None and tree_after.player == 1 - side
            survived = r["collector"] in sim.workers
            c.update(tree_id=tree_id, owner_before=side, age_before=age_before,
                     owner_after=None if tree_after is None else tree_after.player,
                     age_after=None if tree_after is None else tree_after.age,
                     verified_seized=seized, defender_survived=survived,
                     intruder_actual_stay=stayed, sent_collector=screened,
                     intruder_removed=c["intruder"] not in sim.workers)
            counts["sent_to_screen"] += int(screened)
            counts["intercepted_staying_planter"] += int(
                screened and stayed and c["intruder"] not in sim.workers and r["collector"] in sim.workers
            )


def trace_update(digest, obj):
    digest.update((json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode())


def game(seed, side, enabled, pack, policy, weights, sim_type, observer_type):
    from eval_v53 import CFG, _extinct
    import policy_raid_v0_1
    random.seed(seed * 7919 + 104729 * side + 31)
    np.random.seed(seed)
    sim = sim_type(CFG).reset(seed=seed)
    opponent = policy_raid_v0_1.make_agent()
    f = sys.modules["cnet2_feat"].Feat(side)
    observer = observer_type(sim_type)
    counts = Counter()
    key = hashlib.sha256()
    previous = None
    exposures = []
    for t in range(100):
        obs = sim.obs(side)
        inp = f.observe(obs)
        act, logs, reject = choose(f, inp, obs, previous, sim.config, policy, weights, enabled)
        if not enabled:
            if act != pack.agent(obs, sim.config):
                raise AssertionError("baseline does not match public int8 agent")
            counts["matched_pack_calls"] += 1
        actions = [{}, {}]
        actions[side] = act
        actions[1-side] = opponent(sim.obs(1-side), sim.config)
        trace_update(key, [t, actions])
        previous = obs
        observer.step(sim, actions)
        counts["discarded_conflicting_joint_actions"] += int(reject)
        record(logs, reject, enabled, obs, actions, sim, counts)
        if logs:
            exposures.append({"t": t, "rejected": reject, "submitted_actions": actions, "rows": logs})
        if any(_extinct(sim.get_state(), s) for s in (0, 1)):
            break
    return {"steps": t + 1, "cash": float(sim.cash[side]), "opp_cash": float(sim.cash[1-side]),
            "banked": float(observer.d[side]["own_positive_delivery_carbon"]),
            "score_gap_at_stop": float(sim.cash[side]-sim.cash[1-side]), "counts": dict(counts),
            "joint_action_trace_sha256": key.hexdigest(), "exposures": exposures}


def rotate_case(sim, side, rotations):
    # Rotate entities and carbon around the defender's first base, preserving real base locations.
    if not rotations:
        return sim
    pivot = sim.bases[side][0][1]
    row, col = divmod(pivot, 15)
    def rotate(i):
        r, c = divmod(i, 15)
        dr, dc = r-row, c-col
        for _ in range(rotations):
            dr, dc = dc, -dr
        return ((row+dr) % 15) * 15 + (col+dc) % 15
    obs = sim.full_observation()
    carbon = obs["carbon"].copy()
    for i, v in enumerate(carbon):
        obs["carbon"][rotate(i)] = v
    for player in obs["players"]:
        for group in player[2:4]:
            for values in group.values():
                values[0] = rotate(values[0])
    return sim.load_observation(obs)


def fixtures(pack, policy, weights):
    rows = []
    for case, cargo, age, enemy_cash, escort in CASES:
        for side in (0, 1):
            for rotation in range(4):
                sim, tree, _, collector, intruder, _, _, _ = root_position(side, cargo, age, enemy_cash, escort)
                rotate_case(sim, side, rotation)
                state = sim.get_state()
                obs = sim.obs(side)
                row = {"case": case, "seat": side, "quarter_turns": rotation}
                for arm, enabled in (("original", False), ("intent", True)):
                    f = sys.modules["cnet2_feat"].Feat(side)
                    act, logs, reject = choose(f, f.observe(obs), obs, None, sim.config, policy, weights, enabled)
                    if not enabled:
                        pack._AG = None
                        if pack.agent(obs, sim.config) != act:
                            raise AssertionError("constructed baseline disagrees with fresh pack")
                    responses = []
                    for move in MOVES:
                        trial = type(sim)().set_state(state)
                        actions = [{}, {}]
                        actions[side] = act
                        actions[1-side] = {} if move == "NONE" else {intruder: move}
                        trial.step(*actions)
                        surviving_tree = trial.trees.get(tree)
                        responses.append({"intruder_action": move, "intruder_removed": intruder not in trial.workers,
                                          "tree_retained": bool(surviving_tree and surviving_tree.player == side),
                                          "cash": float(trial.cash[side])})
                    row[arm] = {"collector_action": act.get(collector, "NONE"), "actions": act,
                                "rejected": reject, "rows": logs, "responses": responses}
                rows.append(row)
    return rows


def lattice_fixtures(pack, policy, weights):
    """Follow-up to the bank-adjacent fixtures. No change in the frozen scoring rule."""
    from fastsim import FastCarbon
    rows, examples = [], []
    for seat in (0, 1):
        for raw_pos in (15*r+c for r in range(1, 15, 3) for c in range(1, 15, 3)):
            source = FastCarbon().reset(seed=620096100 + seat)
            for direction in range(1, 5):
                pos = int(raw_pos)
                target = int(DEST[pos, direction])
                if pos in source.base_at or target in source.base_at:
                    continue
                state = source.full_observation()
                state["step"] = 80
                own, enemy = state["players"][seat], state["players"][1-seat]
                own[0], enemy[0] = 0, 80
                worker, foe, tree = f"player-{seat}-worker-0", f"player-{1-seat}-worker-0", f"player-{seat}-tree-0"
                own[2][worker] = [pos, 0.0, "COLLECTOR"]
                enemy[2][foe] = [target, 0.0, "PLANTER"]
                own[3][tree] = [target, 9]
                state["trees"][tree] = [None, 0.0]
                sim = FastCarbon().load_observation(state)
                obs = sim.obs(seat)
                f = sys.modules["cnet2_feat"].Feat(seat)
                act, logs, reject = choose(f, f.observe(obs), obs, None, sim.config, policy, weights, True)
                pack._AG = None
                original = pack.agent(obs, sim.config)
                row = {"seat": seat, "cell": pos, "target": target, "adjustments": len(logs),
                       "original": original.get(worker, "NONE"), "adjusted": act.get(worker, "NONE"),
                       "rejected": reject}
                rows.append(row)
                if row["original"] != row["adjusted"]:
                    assert row["adjusted"] == MOVES[direction]
                    if len(examples) < 8:
                        branches = []
                        for move in MOVES:
                            comparison = {}
                            for label, selected in (("original", original), ("adjusted", act)):
                                trial = FastCarbon().set_state(sim.get_state())
                                acts = [{}, {}]
                                acts[seat] = selected
                                acts[1-seat] = {} if move == "NONE" else {foe: move}
                                trial.step(*acts)
                                comparison[label] = {"intruder_removed": foe not in trial.workers,
                                                     "tree_retained": tree in trial.trees and trial.trees[tree].player == seat}
                            branches.append({"reply": move, **comparison})
                        examples.append({"root": row, "logits": logs, "branches": branches})
    return {"scope": "Follow-up mechanism stress test after zero flips in the original bank-adjacent fixtures. "
                     "All valid placements from cells at coordinates 1,4,7,10,13 in each axis, four neighbor directions, "
                     "both seats, excluding base cells. New sparse cold-history fixtures at t80; one zero-cargo collector, "
                     "one enemy planter on our age-9 tree. No retuning or selection of cells by outcome. "
                     "First eight changed positions in traversal order have all five enemy replies replayed.",
            "roots": len(rows), "changed": sum(r["original"] != r["adjusted"] for r in rows),
            "rows": rows, "first_changed_examples": examples}


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--out", type=Path, required=True)
    dest = arg.parse_args().out
    pack = harness.load(harness.PACK, "intent_policy_sapling_int8")
    if pack._OPTS["teacher"] != 2 or any(v for k, v in pack._OPTS.items() if k != "teacher"):
        raise AssertionError("unexpected pack inference options")
    params = pack._dequantize(pack.zlib.decompress(pack.base64.b64decode(pack._W64)), pack._QMETA)
    params["meta"] = np.array(pack._WMETA)
    policy = sys.modules["cnet2_np"].Policy(params)
    weights = np.array(json.loads((HERE / "opponent_intent.json").read_text())["weights"])
    smoke = harness.load(ROOT / "reports/sapling_bootstrap/smoke_ab.py", "intent_policy_observer")
    from fastsim import FastCarbon
    matched = []
    for seed in SEEDS:
        for seat in (0, 1):
            r = {"seed": seed, "seat": seat}
            for arm in ("original", "intent"):
                r[arm] = game(seed, seat, arm == "intent", pack, policy, weights, FastCarbon, smoke.Observer)
            matched.append(r)
    cases = fixtures(pack, policy, weights)
    lattice = lattice_fixtures(pack, policy, weights)
    sums = {}
    for arm in ("original", "intent"):
        counts = Counter()
        for row in matched:
            counts.update(row[arm]["counts"])
        names = ("adjusted_logits", "affected_collector_slots", "local_argmax_flips", "applied_argmax_flips",
                 "sent_to_screen", "intercepted_staying_planter", "discarded_conflicting_joint_actions", "matched_pack_calls")
        sums[arm] = {k: counts[k] for k in names}
        for k in ("steps", "cash", "opp_cash", "banked", "score_gap_at_stop"):
            sums[arm][k + "_sum"] = sum(row[arm][k] for row in matched)
    same_traces = sum(r["original"]["joint_action_trace_sha256"] == r["intent"]["joint_action_trace_sha256"]
                      for r in matched)
    sources = [Path(__file__), harness.PACK, AGENT, HERE / "opponent_intent.json", HERE / "opponent_intent.py",
               HERE / "defense.py", HERE / "defense_probe.py", HERE / "tree_screen_probe.py",
               ROOT / "code/fastsim.py", ROOT / "code/eval_v53.py", Path(smoke.__file__)]
    out = {"formula": "Legal safe reachable target; preaction enemy cash >= seizure price. Add only to its actor logit: 1.5 * P(planter stays | our current/last obs) * min(remaining game steps, live tree life)/lifespan * bank discount. Discount=1/(1+cargo/30) if one step from home and loaded, otherwise 1. Max +1.5 logits. Precommitted before trial.",
           "scope": "Frozen 1200k int8 actor. Hook runs prior to prefix advance; full joint decode falls back on any newly introduced friendly co-arrival or a conflict at the intended tree. Existing unmodified actor conflicts may persist. Observer alone reads submitted opponent actions. Real games stop at t100 or extinction; no full-game utility estimate. Fixtures use fresh history, each opponent reply fixed separately.",
           "sources_sha256": {str(path.relative_to(ROOT)): harness.digest(path) for path in sources},
           "seeds": SEEDS, "matched": matched, "matched_totals": sums, "identical_paired_traces": same_traces,
           "fixtures": cases, "lattice": lattice,
           "fixture_changes": sum(r["original"]["collector_action"] != r["intent"]["collector_action"] for r in cases)}
    with dest.open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
