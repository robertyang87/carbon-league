"""Isolate within-game opponent adaptation from the existing frozen tree-screen scorer.

Run: OPENBLAS_NUM_THREADS=1 python adaptive_game.py --out /tmp/adaptive-game.json
Only the observer receives submitted enemy actions, and only after acting.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import intent_policy_probe as fixed
import numpy as np
from tree_adaptation import AdaptiveTreeIntent, opportunities

SEEDS = tuple(range(620102000, 620102004))
REFERENCE = (620096002, 1)
HORIZON = 100


def encode(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def paired_histories(weights, policy):
    """Construct two legal simulator trajectories with identical states from their second step onward."""
    from fastsim import FastCarbon
    side = 1
    sim = FastCarbon().reset(full_carbon=[0] * 225, cash=0)
    source = sim.full_observation()
    source["step"] = 68
    mine, opponent = source["players"][side], source["players"][1-side]
    opponent[0] = 30
    locations = [15*r+c for r in (2, 11) for c in (2, 8)]
    foes = []
    for i, tile in enumerate(locations):
        collector = f"player-{side}-worker-{i}"
        intruder = f"player-{1-side}-worker-{i}"
        tree = f"player-{side}-tree-{i}"
        mine[2][collector] = [int(fixed.DEST[tile, 4]), 0.0, "COLLECTOR"]
        mine[3][tree] = [tile, 8]
        source["trees"][tree] = [None, 0.0]
        opponent[2][intruder] = [tile, 0.0, "PLANTER"]
        foes.append(intruder)
    bases = sorted(opponent[1])
    bank = opponent[1][bases[1]]
    bank_uid = f"player-{1-side}-worker-4"
    opponent[2][bank_uid] = [int(fixed.DEST[bank, 4]), 100, "COLLECTOR"]
    memories, logs, input_arrays = {}, {}, {}
    for style in ("four_stays", "four_retreats"):
        state = FastCarbon().load_observation(copy.deepcopy(source))
        f = sys.modules["cnet2_feat"].Feat(side)
        memory = AdaptiveTreeIntent()
        previous = None
        seen = []
        for i in range(13):
            obs = state.obs(side)
            seen.append(encode(obs))
            memory.observe(obs, state.config)
            inputs = f.observe(obs)
            if i == 12:
                break
            enemy = {}
            # Initial shared budget covers seizure in the pre-action observation.
            # Spending the 30 on one recruit (common to both branches) precedes
            # tree resolution, so the four stay/retreat choices leave all tree owners unchanged.
            if i == 0:
                enemy[bases[0]] = "RECCOLLECTOR"
                if style == "four_retreats":
                    enemy.update({uid: "RIGHT" for uid in foes})
            elif i == 1:
                if style == "four_retreats":
                    enemy.update({uid: "LEFT" for uid in foes})
            elif i == 2:
                enemy.update({uid: "RIGHT" for uid in foes})
                enemy[bank_uid] = "RIGHT"
            else:
                direction = "LEFT" if i == 11 or i % 2 == 0 else "RIGHT"
                enemy.update({uid: direction for uid in foes})
            actions = [{}, {}]
            actions[1-side] = enemy
            previous = obs
            state.step(*actions)
        if memory.learned != 4 or memory.censored:
            raise AssertionError((style, memory.learned, memory.censored))
        frozen = fixed.predict(obs, weights, previous)
        combined = memory.probabilities(obs, state.config, frozen)
        act, candidates, reject = fixed.choose(f, inputs, obs, previous, state.config, policy, weights, True,
                                              intent_filter=memory.probabilities)
        logs[style] = {"observation_hashes": seen, "current_engine_state": encode(state.get_state()),
                       "resolved_labels": memory.learned, "decayed_counts": memory.counts.tolist(),
                       "per_intruder_p_stay": {uid: [float(frozen[uid][0]), float(combined[uid][0])] for uid in foes},
                       "actor_action": act, "candidates": candidates, "joint_rejected": reject}
        input_arrays[style] = inputs
        memories[style] = memory
    left, right = (logs[s] for s in ("four_stays", "four_retreats"))
    if left["observation_hashes"][1] == right["observation_hashes"][1]:
        raise ValueError("Expected different public observations after the first move")
    if (left["observation_hashes"][2:] != right["observation_hashes"][2:]
            or left["current_engine_state"] != right["current_engine_state"]):
        raise ValueError("Histories did not converge")
    for key, x in input_arrays["four_stays"].items():
        if not np.array_equal(x, input_arrays["four_retreats"][key]):
            raise ValueError(f"Actor state depends on different histories: {key}")
    return memories, {
        "design": "Synthetic t68 scenario with four distinct funded enemy planters on our live trees and four adjacent defenders. Enemy also recruits that turn; the money is spent before capture. Four stay versus four step east then return. At t70 the worlds reconverge. The identical enemy collector later deposits, all four planters oscillate away then return. At t80 full engine states, last 11 observations and every actor input array exactly match. All transitions produced by FastCarbon. The wagers help separate intended stays from successful captures.",
        "conditioned_action_changed": left["actor_action"] != right["actor_action"],
        "by_history": logs,
    }


def run_game(seed, side, adaptive, policy, weights, observer_type, historical=None):
    from eval_v53 import CFG, _extinct
    from fastsim import FastCarbon
    import policy_raid_v0_1
    random.seed(seed*7919 + 104729*side + 31)
    np.random.seed(seed)
    sim = FastCarbon(CFG).reset(seed=seed)
    opponent = policy_raid_v0_1.make_agent()
    features = sys.modules["cnet2_feat"].Feat(side)
    observer = observer_type(FastCarbon)
    memory = AdaptiveTreeIntent()
    previous = None
    trace = hashlib.sha256()
    counts = Counter()
    exposures, crossed = [], {}
    for t in range(HORIZON):
        obs = sim.obs(side)
        memory.observe(obs, sim.config)
        inp = features.observe(obs)
        action, logs, reject = fixed.choose(features, inp, obs, previous, sim.config, policy, weights, True,
                                            intent_filter=memory.probabilities if adaptive else None)
        counts["eligible_worker_screens"] += sum(len(r["candidates"]) for r in logs)
        counts["funded_raider_opportunities"] += len(opportunities(obs, sim.config))
        if logs:
            other, other_logs, _ = fixed.choose(features, inp, obs, previous, sim.config, policy, weights, True,
                                                intent_filter=None if adaptive else memory.probabilities)
            counts["on_state_joint_differences"] += other != action
            for actual, contrast in zip(logs, other_logs):
                for chosen, alt in zip(actual["candidates"], contrast["candidates"]):
                    chosen["comparison_other_p_stay"] = alt["stay_p"]
        if historical is not None and t in (82, 83):
            # This cross-over is intentionally NOT claimed to be a possible single
            # natural match. Each synthetic history is independently legal; both
            # feed the same last two archived observations to the same actor state.
            for name, recalled in historical.items():
                recalled.observe(obs, sim.config)
                if t == 83:
                    candidate, candidate_logs, fallback = fixed.choose(
                        features, inp, obs, previous, sim.config, policy, weights, True,
                        intent_filter=recalled.probabilities)
                    crossed[name] = {"actions": candidate, "rows": candidate_logs, "fallback": fallback}
        enemy_action = opponent(sim.obs(1-side), sim.config)
        acts = [{}, {}]
        acts[side], acts[1-side] = action, enemy_action
        fixed.trace_update(trace, [t, acts])
        if historical is not None and t == 83:
            tree = "player-1-tree-2"
            if sim.trees[tree].pos != 150 or sim.trees[tree].player != side:
                raise ValueError("Historic tree/seat identity changed")
            for case in crossed.values():
                counterfactual = FastCarbon(CFG).set_state(sim.get_state())
                replies = list(acts)
                replies[side] = case["actions"]
                counterfactual.step(*replies)
                case["one_step_fixed_enemy_reply"] = {
                    "id": tree, "cell": 150, "owner_before": side,
                    "owner_after": counterfactual.trees[tree].player,
                    "defender_survived": "player-1-worker-0" in counterfactual.workers,
                    "raider_survived": "player-0-worker-2" in counterfactual.workers,
                }
        observer.step(sim, acts)
        fixed.record(logs, reject, True, obs, acts, sim, counts)
        if historical is not None and t == 83:
            for name, case in crossed.items():
                case["actual_intruder_action_in_reference"] = enemy_action.get("player-0-worker-2", "NONE")
        if logs:
            exposures.append({"t": t, "sampled_own_joint": action, "rows": logs})
        previous = obs
        if any(_extinct(sim.get_state(), seat) for seat in (0, 1)):
            break
    result = {"joint_trace_sha256": trace.hexdigest(), "steps": t+1, "cash": sim.cash[side],
              "opponent_cash": sim.cash[1-side], "cash_gap_at_stop": sim.cash[side]-sim.cash[1-side],
              "banked": observer.d[side]["own_positive_delivery_carbon"], "counts": dict(counts),
              "resolutions_from_consecutive_observations": memory.learned, "censored": memory.censored,
              "screen_events": exposures}
    return result, crossed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new output path")
    pack = fixed.harness.load(fixed.harness.PACK, "adaptive_probe_sapling")
    if pack._OPTS["teacher"] != 2 or any(v for k, v in pack._OPTS.items() if k != "teacher"):
        raise ValueError("Unexpected actor configuration")
    params = pack._dequantize(pack.zlib.decompress(pack.base64.b64decode(pack._W64)), pack._QMETA)
    params["meta"] = np.array(pack._WMETA)
    policy = sys.modules["cnet2_np"].Policy(params)
    weights_path = HERE.parent / "opponent_intent.json"
    weights = np.array(json.loads(weights_path.read_text())["weights"])
    smoke = fixed.harness.load(fixed.ROOT/"reports/sapling_bootstrap/smoke_ab.py", "adaptive_probe_observer")
    styles, convergence = paired_histories(weights, policy)
    reference, crossed = run_game(*REFERENCE, False, policy, weights, smoke.Observer, styles)
    old_path = HERE.parent / "intent_policy_probe.json"
    prior = json.loads(old_path.read_text())
    old = next(r["intent"] for r in prior["matched"] if (r["seed"], r["seat"]) == REFERENCE)
    if reference["joint_trace_sha256"] != old["joint_action_trace_sha256"]:
        raise ValueError("Default scorer no longer reproduces the prior reference")
    pairs = []
    for seed in SEEDS:
        for side in (0, 1):
            pair = {"seed": seed, "seat": side}
            for adaptive, name in ((False, "fixed"), (True, "adaptive")):
                pair[name], _ = run_game(seed, side, adaptive, policy, weights, smoke.Observer)
            pairs.append(pair)
    result = {
        "scope": "Frozen actor, same frozen prior/logit bonus in both arms; only the online update differs. Fresh maps four × both seats, reactive raid opponent, to 100 steps or extinction. One known historical case and a constructed convergence test are separate from this sample.",
        "sources_sha256": {str(p.relative_to(fixed.ROOT)): fixed.harness.digest(p) for p in
                           (Path(__file__), HERE/"tree_adaptation.py", Path(fixed.__file__), weights_path,
                            fixed.harness.PACK, fixed.AGENT, fixed.ROOT/"code/fastsim.py", old_path)},
        "seeds": SEEDS, "pairs": pairs,
        "identical_joint_traces": sum(p["fixed"]["joint_trace_sha256"] == p["adaptive"]["joint_trace_sha256"] for p in pairs),
        "means": {arm: {k: sum(p[arm][k] for p in pairs)/len(pairs)
                        for k in ("cash", "opponent_cash", "cash_gap_at_stop", "banked")}
                  for arm in ("fixed", "adaptive")},
        "convergent_fixture": convergence,
        "cross_over": {"scope": "Each memory is primed by its legal t68–80 synthetic history, then separately supplied t82 and t83 of the same historical match. This joins two scenario histories and is not one natural trajectory. Actor tensors, previous obs and contemporaneous board, opponents and actual reply are identical. Historical matchup is excluded from fresh A/B.",
                       "seed": REFERENCE[0], "seat": REFERENCE[1], "t": 83,
                       "reference_trace": reference["joint_trace_sha256"],
                       "fixed_reference_screens": [r for r in reference["screen_events"] if r["t"] == 83],
                       "by_history": crossed},
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
