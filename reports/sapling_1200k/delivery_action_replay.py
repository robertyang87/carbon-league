"""Reproduce each recorded candidate action and evaluate same-root parent alternatives."""
import argparse
from collections import Counter, defaultdict
import copy
import gzip
import json
from pathlib import Path

from tree_audited_game import exam, prior
from tree_win_gate import validate_game, verify_raw
from turning_point_ledger import CashEvents, distances, write


def effective(actions):
    return {unit: command for unit, command in actions.items() if command is not None}


def describe(sim, unit, parent, candidate, distance, seat):
    worker = sim.workers.get(unit)
    detail = {"id": unit, "parent": parent, "candidate": candidate}
    if worker is None:
        if unit not in dict(sim.bases[seat]):
            raise ValueError(f"unknown unit {unit}")
        return {**detail, "role": "base"}
    homes = [pos for _, pos in sim.bases[seat]]

    def distance_home(command):
        pos = worker.pos if command is None else sim._move_nbr[command][worker.pos]
        return min(distance[pos][b] for b in homes)

    return {**detail, "role": "collector" if worker.is_collector else "planter",
            "start": worker.pos, "cargo": worker.carbon,
            "distance_home_before": distance_home(None),
            "distance_home_parent": distance_home(parent),
            "distance_home_candidate": distance_home(candidate)}


def cash_outcome(sim, seat):
    return {"cash": [sim.cash[seat], sim.cash[1-seat]],
            "deliveries": [sum(e[2] for e in sim.cash.events
                               if e[0] == "own_delivery" and e[1] == p)
                           for p in (seat, 1-seat)],
            "cargo": [sum(sim.workers[k].carbon for k in sim.player_workers[p])
                      for p in (seat, 1-seat)],
            "workers": [len(sim.player_workers[p]) for p in (seat, 1-seat)]}


def advance(sim, joint, seat, events):
    before = set(sim.player_workers[seat])
    sim.step(*joint)
    for kind, owner, value, *details in sim.cash.events:
        if kind == "own_delivery" and owner == seat and value > 0:
            events[details[0]].append({"turn_end": sim.step_count, "event": "deposit", "cash": value})
    for unit in before.difference(sim.player_workers[seat]):
        events[unit].append({"turn_end": sim.step_count, "event": "death"})


def inspect(run, row, header, candidate, parent, ablated):
    validate_game(row)
    verify_raw(run, row, header["pack_sha256"], header["weight_sha256"])
    raw = json.loads(gzip.decompress((run / row["raw_file"]).read_bytes()))
    seed, seat = row["seed"], row["candidate_seat"]
    sim = exam.FastCarbon(raw["game_configuration"]).reset(seed=seed)
    sim.cash = CashEvents(sim.cash)
    distance = distances(sim.size)
    changed = []
    student_sensitive = []
    unit_events = defaultdict(list)
    unit_totals = Counter()
    for step, joint in enumerate(raw["actions"]):
        observed = sim.obs(seat)
        executed = candidate.agent(observed, sim.config)
        if executed != joint[seat]:
            raise ValueError(f"deployed actions differ from recorded at {seed}/{seat}/{step}")
        base_actions = effective(parent.agent(sim.obs(seat), sim.config))
        actual_actions = effective(executed)
        no_student = effective(ablated.agent(sim.obs(seat), sim.config))
        different_ids = sorted(k for k in actual_actions.keys() | no_student.keys()
                               if no_student.get(k) != actual_actions.get(k))
        if different_ids:
            student_sensitive.append([step, different_ids])
        # Check claimed freight interventions against the independently re-run parent,
        # then verify the actual deployed label for each explicitly overridden slot.
        claims = candidate._FREIGHT_LAST
        if bool(claims) != (base_actions != actual_actions):
            raise ValueError(f"intervention attribution disagrees at {seed}/{seat}/{step}")
        if claims:
            labels = candidate._FEAT.labels(executed)
            if any(int(labels[slot]) != chosen or old == chosen for slot, old, chosen in claims):
                raise ValueError(f"executed labels disagree at {seed}/{seat}/{step}")
        sim.cash.events = []
        if base_actions == actual_actions:
            advance(sim, joint, seat, unit_events)
            continue
        units = [describe(sim, unit, base_actions.get(unit), actual_actions.get(unit), distance, seat)
                 for unit in sorted(base_actions.keys() | actual_actions.keys())
                 if base_actions.get(unit) != actual_actions.get(unit)]
        unit_totals.update(unit["role"] for unit in units)
        alternative = copy.deepcopy(sim)
        alt_joint = list(joint)
        alt_joint[seat] = base_actions
        alternative.step(*alt_joint)
        advance(sim, joint, seat, unit_events)
        changed.append({"step": step, "units": units,
                        "direct_overrides": [{"id": candidate._FEAT.own_uid[slot],
                                              "prefix_proposal": old, "selected": chosen}
                                             for slot, old, chosen in claims],
                        "parent_one_step": cash_outcome(alternative, seat),
                        "candidate_one_step": cash_outcome(sim, seat)})
    if prior.hashed(sim.get_state()) != row["final_state_sha256"]:
        raise ValueError(f"terminal state differs from recorded at {seed}/{seat}")
    for item in changed:
        cap = min(item["step"] + 20, len(raw["actions"]))
        for override in item["direct_overrides"]:
            following = [event for event in unit_events[override["id"]]
                         if item["step"] < event["turn_end"] <= cap]
            override["first_actual_deposit_or_death"] = following[0] if following else None
            override["observed_through_turn_end"] = cap
    stats = copy.deepcopy(getattr(candidate, "_COUNTS", {}))
    runtime_stats = dict(candidate._FREIGHT_COUNTS)
    if runtime_stats.get("changed_roots", 0) != len(changed):
        raise ValueError("end-of-game runtime attribution differs")
    return {"seed": seed, "seat": seat, "raw_sha256": row["raw_sha256"],
            "verified_roots": len(raw["actions"]), "changed_roots": len(changed),
            "changed_units_by_role": dict(unit_totals),
            "pack_reported_policy_head_choices": stats, "freight_counts": runtime_stats,
            "student_sensitive_roots": len(student_sensitive),
            "student_sensitive_steps_and_units": student_sensitive, "roots": changed}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "candidate", "parent", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("--parent-sha256", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    header_file = a.run / "results.json"
    report = json.loads(header_file.read_bytes())
    header = report["header"]
    if prior.digest(a.manifest) != header["weight_sha256"]:
        raise ValueError("frozen manifest does not match the game")
    sources = json.loads(a.manifest.read_bytes())["files_sha256"]
    repo = Path(__file__).resolve().parents[2]
    for path, checksum in sources.items():
        if prior.digest(repo / path) != checksum:
            raise ValueError(f"deployed source mismatch: {path}")
    if (sources[a.candidate.resolve().relative_to(repo).as_posix()] != header["pack_sha256"]
            or sources[a.parent.resolve().relative_to(repo).as_posix()] != a.parent_sha256):
        raise ValueError("these actors do not belong to the frozen recipe")
    keys = [(g["seed"], g["candidate_seat"]) for g in report["games"]]
    expected = {(seed, seat) for seed in header["seeds"] for seat in (0, 1)}
    if (report["status"] != "complete" or len(keys) != len(expected) or set(keys) != expected
            or prior.digest(a.candidate) != header["pack_sha256"]
            or prior.digest(a.parent) != a.parent_sha256):
        raise ValueError("need a complete run and matching arm identities")
    # These modules own their own recurrent features. Neither selector sees the other arm's moves.
    candidate = prior.shared.module(a.candidate, "candidate_evaluated_pack")
    parent = prior.shared.module(a.parent, "frozen_reference_pack")
    ablated = prior.shared.module(a.candidate, "counterfactual_zero_sapling_logit")
    if ablated._HYBRID["student_weight"] != .25 or ablated._HYBRID["expert_weight"] != .75:
        raise ValueError("unexpected blend coefficients")
    ablated._HYBRID["student_weight"] = 0.
    data = {"sapling_ablation": "Same deployed selector, freight logic and 0.75 expert coefficient. "
                               "Set only the 0.25 Sapling multiplier to zero; count effective action changes "
                               "on the same observations. This measures decision influence, not win-rate uplift.",
            "scope": "On each candidate trajectory, compare actually executed actions to a persistent "
                     "parent on the same visible observation history. For changed roots replay one alternative "
                     "joint turn with the recorded opponent action fixed. Economy arrays are [ours, opponent]. "
                     "For directly rerouted workers, record their first actual deposit or death in the next "
                     "20 transitions or up to termination. A deposit can follow multiple decisions by the same "
                     "worker; these events cannot be summed as distinct receipts. One-step outcomes do not "
                     "estimate the value of a different subsequent trajectory.",
            "candidate_pack_sha256": header["pack_sha256"], "parent_pack_sha256": a.parent_sha256,
            "input_results_sha256": prior.digest(header_file), "manifest_sha256": header["weight_sha256"],
            "games": [inspect(a.run, row, header, candidate, parent, ablated) for row in report["games"]]}
    write(a.out, data, compressed=a.out.suffix == ".gz")


if __name__ == "__main__":
    main()
