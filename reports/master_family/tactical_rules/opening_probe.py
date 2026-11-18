"""Test the Top-5 common first-turn recruit pattern in the public Sapling int8 agent.

Both arms call the actual package. A scoped hook inside its decoder can change
only the two base actions at the standard opening. It is removed before the
opponent or the next step acts. Run from the repository root:
  python reports/master_family/tactical_rules/opening_probe.py --out /tmp/opening.json
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
from unittest.mock import patch

from defense_probe import PACK, ROOT, RULE_OPPONENT, digest, load

SEEDS = tuple(range(620098000, 620098004))
T = 60


def opening_action(pack, obs, cfg):
    cash, bases, workers, _ = obs["players"][obs["player"]]
    eligible = (
        obs["step"] == 0 and cash == 80 and not workers and len(bases) == 2
        and cfg["workerLimit"] >= 2 and cfg["recCollectorCost"] == 30
        and cfg["recPlanterCost"] == 30 and cfg["recCollectorIncreaseCost"] == 0
        and cfg["recPlanterIncreaseCost"] == 0
    )
    if not eligible:
        return pack.agent(obs, cfg)
    policy_class = sys.modules["cnet2_np"].Policy
    original_act = policy_class.act
    choices = []

    def constrain(slot, logits, prefix, greedy):
        # The embedded Policy.act advances Prefix AND its recurrent action
        # embedding using the action returned here before decoding the next slot.
        if slot != 10 + len(choices) or slot > 11 or min(logits[1], logits[2]) < -1e8:
            raise AssertionError("expected two legal, sequential base slots")
        # Preserve greedy if it buys; otherwise choose the better of its legal
        # role logits. If the first base chose planter, the second must collect.
        action = 1 if slot == 11 and choices[0] == 2 else 1 + int(logits[2] > logits[1])
        choices.append(action)
        return action

    def first_turn(self, *args, **kwargs):
        if kwargs.get("hook") is not None:
            raise AssertionError("Unexpected production hook")
        return original_act(self, *args, **dict(kwargs, hook=constrain))

    with patch.object(policy_class, "act", first_turn):
        action = pack.agent(obs, cfg)
    if len(choices) != 2 or 1 not in choices:
        raise AssertionError("opening contract did not execute")
    return action


def run_arm(seed, seat, enabled, pack, simulator, smoke):
    random.seed(seed * 7919 + seat * 104729 + 31)
    sim = simulator.FastCarbon().reset(seed=seed)
    observer = smoke.Observer(simulator.FastCarbon)
    opponent = load(RULE_OPPONENT, "opening_probe_opponent").make_agent()
    calls = [0, 0]
    snapshots = {}
    opening = None
    terminal = False
    while sim.step_count < T and not terminal:
        actions = []
        for side in (0, 1):
            observation = sim.obs(side)
            if side != seat:
                action = opponent(observation, sim.config)
            elif sim.step_count == 0 and enabled:
                action = opening_action(pack, observation, sim.config)
            else:
                action = pack.agent(observation, sim.config)
            if side == seat and sim.step_count == 0:
                bases = sorted(observation["players"][side][1])
                opening = [action.get(base, "NONE") for base in bases]
            calls[side] += 1
            actions.append(action)
        observer.step(sim, actions)
        terminal = any(not sim.player_workers[p] and not sim.player_trees[p] and
                       (not sim.bases[p] or sim.cash[p] < min(sim.config["recCollectorCost"],
                                                               sim.config["recPlanterCost"]))
                       for p in (0, 1))
        if sim.step_count in (30, T) or terminal:
            snapshots[str(sim.step_count)] = smoke.snapshot(sim, observer, seat, [0, 0], calls)
    final_hash = hashlib.sha256(json.dumps(sim.get_state(), sort_keys=True).encode()).hexdigest()
    return {"seed": seed, "seat": seat, "opening_by_base_id": opening,
            "steps": sim.step_count, "terminal": terminal, "snapshots": snapshots,
            "final_simulation_state_sha256": final_hash}


def summarize(rows, arm):
    relevant = [r[arm]["snapshots"][str(T)] for r in rows if str(T) in r[arm]["snapshots"]]
    return {
        "appearances": len(rows), "reaching_t60": len(relevant),
        "opening_pairs": dict(Counter("+".join(r[arm]["opening_by_base_id"]) for r in rows)),
        "completed_loops_at_t60": sum(r["completed_collection_delivery_loop"] for r in relevant),
        "delivery_carbon_sum_at_t60": sum(r["own_positive_delivery_carbon"] for r in relevant),
        "mean_cash_at_t60": statistics.mean(r["cash"] for r in relevant) if relevant else None,
        "mean_live_trees_at_t60": statistics.mean(r["alive_trees"] for r in relevant) if relevant else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    pack = load(PACK, "opening_probe_sapling")
    if pack._OPTS.get("teacher") != 2 or any(v for k, v in pack._OPTS.items() if k != "teacher"):
        raise ValueError("Unexpected Sapling options")
    simulator = load(ROOT / "code/fastsim.py", "opening_probe_fastsim")
    smoke = load(ROOT / "reports/sapling_bootstrap/smoke_ab.py", "opening_probe_accounting")
    rows = []
    for seed in SEEDS:
        for seat in (0, 1):
            a = run_arm(seed, seat, False, pack, simulator, smoke)
            b = run_arm(seed, seat, True, pack, simulator, smoke)
            changed = a["opening_by_base_id"] != b["opening_by_base_id"]
            if not changed and a["final_simulation_state_sha256"] != b["final_simulation_state_sha256"]:
                raise ValueError("Equal opening produced different paired trajectory")
            rows.append({"original": a, "constrained": b, "opening_changed": changed})
    result = {
        "scope": "Four new maps, both seats, one responsive rule opponent, 60 transitions. No terminal strength claim.",
        "pack_sha256": digest(PACK), "opponent_sha256": digest(RULE_OPPONENT),
        "fastsim_sha256": digest(ROOT / "code/fastsim.py"),
        "observer_sha256": digest(ROOT / "reports/sapling_bootstrap/smoke_ab.py"),
        "seeds": SEEDS, "horizon": T, "changed_openings": sum(row["opening_changed"] for row in rows),
        "summary": {arm: summarize(rows, arm) for arm in ("original", "constrained")},
        "pairs": rows,
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
