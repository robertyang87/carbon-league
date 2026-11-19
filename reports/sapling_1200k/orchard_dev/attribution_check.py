"""Check real joint actions and same-observation removal of each candidate component."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
sys.path.insert(0, str(RUN.parent))
import tree_audited_game as arena


def differing_orders(actual, alternative):
    return {unit: [actual.get(unit), alternative.get(unit)]
            for unit in sorted(actual.keys() | alternative.keys())
            if actual.get(unit) != alternative.get(unit)}


def purchase_receipts(steps, seat, changes):
    """Follow each newly encouraged purchase through first ownership tenure."""
    receipts = []
    for root in changes:
        for slot, _, after in root["accepted_orchard_changes"]:
            if after != 0:
                continue
            turn = root["step"]
            worker = root["changed_slot_ids"][str(slot)]
            cash = steps[turn]["cash_events"]
            expense = [e for e in cash if e[0] in ("plant", "seize") and e[1] == seat and e[3] == worker]
            item = {"decision_step": turn, "unit": worker, "settled": bool(expense)}
            receipts.append(item)
            if not expense:
                continue
            if len(expense) != 1:
                raise ValueError("ambiguous orchard investment")
            kind, _, debit, _, pos = expense[0]
            tree, = [tid for action, tid, cell, _, new in steps[turn]["tree_changes"]
                     if action in ("new", "seizure") and cell == pos and new == seat]
            item.update(kind=kind, tree=tree, cost=-debit, absorption=0., collision_cash=0.)
            item["end_reason"] = "terminal"
            for row in steps[turn:]:
                for event in row["cash_events"]:
                    if event[1] != seat:
                        continue
                    if event[0] == "tree_absorption" and event[3] == tree:
                        item["absorption"] += event[2]
                    elif event[0] == "tree_collision_cash" and event[5] == tree:
                        item["collision_cash"] += event[2]
                item["end_state"] = row["t"]
                endings = [action for action, tid, _, old, new in row["tree_changes"]
                           if tid == tree and old == seat and new != seat]
                if endings:
                    item["end_reason"] = endings[0]
                    break
            item["direct_net"] = item["absorption"] + item["collision_cash"] - item["cost"]
    return receipts


def audit(row, manifest):
    prior = arena.prior
    tape = RUN / row["raw_file"]
    if prior.digest(tape) != row["raw_sha256"]:
        raise ValueError(f"action tape identity mismatch: {tape}")
    raw = json.loads(gzip.decompress(tape.read_bytes()))
    pack_dir = ROOT / "data/weights/sapling_1200k"
    candidate_path = pack_dir / "orchard_it50_v3.py"
    parent_path = pack_dir / "hybrid_it50_v3.py"
    for path in (candidate_path, parent_path):
        if prior.digest(path) != manifest["files_sha256"][str(path.relative_to(ROOT))]:
            raise ValueError(f"model identity mismatch: {path}")
    if prior.digest(candidate_path) != raw["pack_sha256"]:
        raise ValueError("archived game used a different candidate")
    actual = prior.shared.module(candidate_path, "replay_candidate")
    no_student = prior.shared.module(candidate_path, "replay_without_student")
    # The orchard action rule uses absolute logit gaps, so retain the exact
    # original expert scale when removing only the student's contribution.
    no_student._HYBRID["student_weight"] = 0.
    no_orchard = prior.shared.module(parent_path, "replay_without_orchard")
    seed, seat = row["seed"], row["candidate_seat"]
    sim = arena.exam.FastCarbon(raw["game_configuration"]).reset(seed=seed)
    roots, total = [], Counter()
    for turn, commands in enumerate(raw["actions"]):
        observation = sim.obs(seat)
        played = actual.agent(observation, sim.config)
        if played != commands[seat]:
            raise ValueError(f"candidate differs at {seed}/{seat}/{turn}")
        alternatives = {
            "remove_sapling": no_student.agent(observation, sim.config),
            "remove_orchard": no_orchard.agent(observation, sim.config),
        }
        changes = {name: differing_orders(played, other) for name, other in alternatives.items()}
        for name, diffs in changes.items():
            total[name + "_changed_roots"] += bool(diffs)
            total[name + "_changed_orders"] += len(diffs)
        if any(changes.values()) or actual._ORCHARD_LAST:
            roots.append({"step": turn, "accepted_orchard_changes": actual._ORCHARD_LAST,
                          "changed_slot_ids": {str(slot): actual._FEAT.own_uid[slot]
                                               for slot, _, _ in actual._ORCHARD_LAST},
                          "orders_actual_then_ablation": changes})
        sim.step(*commands)
    if sim.step_count != row["actual_transitions"] or prior.hashed(sim.get_state()) != row["final_state_sha256"]:
        raise ValueError(f"terminal state mismatch: {seed}/{seat}")
    detail_file = RUN / "lead_ledger" / f"{RUN.name}_{seed}_{seat}.json.gz"
    detail = json.loads(gzip.decompress(detail_file.read_bytes()))
    if detail["terminal_state_sha256"] != row["final_state_sha256"]:
        raise ValueError("wrong cash ledger for action trajectory")
    return {"seed": seed, "seat": seat, "matched_steps": sim.step_count, "totals": total,
            "archived_raw_sha256": row["raw_sha256"], "mixed_slot_counts": actual._COUNTS,
            "orchard_counts": actual._ORCHARD_COUNTS, "changed_roots": roots,
            "encouraged_purchase_factual_receipts": purchase_receipts(detail["steps"], seat, roots)}


if __name__ == "__main__":
    report = json.loads((RUN / "results.json").read_bytes())
    manifest = json.loads((RUN / "build_manifest.json").read_bytes())
    expected = {(seed, seat) for seed in (620224310, 620224311) for seat in (0, 1)}
    if report["status"] != "complete" or len(report["games"]) != len(expected) or {
        (row["seed"], row["candidate_seat"]) for row in report["games"]
    } != expected:
        raise ValueError("incomplete map and seat set")
    rows = [audit(row, manifest) for row in report["games"]]
    print(json.dumps({"branch": "same recorded observation history and decision root, real future follows archived joint commands",
                      "remove_sapling": "expert weight fixed at .75, student weight 0, same orchard adjustment",
                      "remove_orchard": "unchanged original mixture with expert weight .75 and Sapling .25",
                      "games": rows}, separators=(",", ":")))
