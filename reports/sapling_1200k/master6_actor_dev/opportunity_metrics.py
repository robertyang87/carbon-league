"""Compare the two frozen quantized actors on the same Master6 holdout prefixes and tactical opportunities."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from train import HERE, ROOT, WEIGHTS, digest, model, predict, public, quantized, write_json


def openings(data, rows, slots):
    """Return the public opportunity and the moves that qualify as the tactical response."""
    moves = data["mf"][rows, slots]
    legal = data["legal"][rows, slots]
    kinds = data["tok_kind"][rows, slots]
    cargo = data["tok_cargo"][rows, slots]
    collector = (slots < 10) & (kinds == 0)
    planter = (slots < 10) & (kinds == 1)
    danger = moves[:, :, 1] > 0
    safe = legal & ~danger
    bank = legal & (moves[:, :, 5] > 0)
    heavier_foe = legal & (moves[:, :, 2] > 0)
    enemy_tree = legal & (moves[:, :, 9] > 0)
    stay = np.zeros_like(legal)
    stay[:, 0] = True
    return {
        "loaded_safe_escape": (collector & (cargo >= 30) & danger[:, 0] & safe.any(1), safe),
        "loaded_bank": (collector & (cargo >= 30) & bank.any(1), bank),
        "approach_grabbable_enemy_cargo": (collector & heavier_foe.any(1), heavier_foe),
        "planter_move_onto_enemy_tree": (planter & ~enemy_tree[:, 0] & enemy_tree.any(1), enemy_tree),
        "planter_stay_on_enemy_tree": (planter & enemy_tree[:, 0], stay),
        "protect_owned_tree": (planter & (moves[:, 0, 8] > 0) & (moves[:, 0, 4] > 0), stay),
        "base_recruit_choice": (slots >= 10, legal & ~stay),
    }


def accuracy(truth, parent, student):
    """A correction fixes an old disagreement; a damaged slot was previously correct."""
    n = len(truth)
    a = parent == truth
    b = student == truth
    corrected = int((~a & b).sum())
    damaged = int((a & ~b).sum())
    return {
        "slots": n, "parent_correct": int(a.sum()), "student_correct": int(b.sum()),
        "parent_accuracy": float(a.mean()) if n else None, "student_accuracy": float(b.mean()) if n else None,
        "corrected": corrected, "damaged": damaged, "net_correct": corrected-damaged,
        "prediction_changed": int((parent != student).sum()),
    }


def summarize(mask, response, rows, gids, seats, target, parent, student):
    select = np.flatnonzero(mask)
    expert_did = response[select, target[select]]
    parent_did = response[select, parent[select]]
    student_did = response[select, student[select]]
    exact = accuracy(target[select], parent[select], student[select])
    qualified = select[expert_did]
    declined = select[~expert_did]
    maps = np.unique(gids[select])
    by_map = {}
    for g in maps:
        ix = select[gids[select] == g]
        by_map[str(g)] = accuracy(target[ix], parent[ix], student[ix])
    by_seat = {}
    for seat in (0, 1):
        ix = select[seats[select] == seat]
        by_seat[str(seat)] = accuracy(target[ix], parent[ix], student[ix])
    return {
        "opportunities": len(select), "distinct_roots": len(np.unique(rows[select])), "distinct_maps": len(maps),
        "expert_used_response": int(expert_did.sum()), "parent_used_response": int(parent_did.sum()),
        "student_used_response": int(student_did.sum()),
        "exact_action": exact,
        "exact_when_expert_used_response": accuracy(target[qualified], parent[qualified], student[qualified]),
        "exact_when_expert_declined_response": accuracy(target[declined], parent[declined], student[declined]),
        "response_level": accuracy(expert_did, parent_did, student_did),
        "parent_response_on_expert_yes": int(parent_did[expert_did].sum()),
        "student_response_on_expert_yes": int(student_did[expert_did].sum()),
        "parent_response_on_expert_no": int(parent_did[~expert_did].sum()),
        "student_response_on_expert_no": int(student_did[~expert_did].sum()),
        "maps_exact_gain_loss_tie": dict(Counter("gain" if x["net_correct"] > 0 else "loss"
                                               if x["net_correct"] < 0 else "tie" for x in by_map.values())),
        "exact_by_map": by_map, "exact_by_physical_seat": by_seat,
        "low_support": len(select)<30 or min(len(qualified), len(declined))<5 or len(maps)<5,
    }


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    split_file = HERE / "data_split.json"
    receipts_file = HERE / "data_receipts.json.gz"
    weights_file = HERE / "weight_receipt.json"
    audit_file = HERE / "data_audit.json"
    metrics_file = HERE / "metrics.json"
    split_raw = split_file.read_bytes()
    cards = [g for g in json.loads(split_raw)["games"] if g["split"] == "holdout"]
    if len(cards) != 18 or len({g["terrain_sha256"] for g in cards}) != len(cards):
        raise ValueError("expected the original eighteen-map holdout")
    receipts = json.loads(gzip.decompress(receipts_file.read_bytes()))
    weights = json.loads(weights_file.read_bytes())
    if (receipts["split_sha256"] != digest(split_raw) or weights["recipe"]["split_sha256"] != digest(split_raw)
            or weights["recipe"]["parent_sha256"] != "df0cf175db407712f31c188184c1395c0ac9b9c26ddc6b37b9dacda9fc43e5be"):
        raise ValueError("mismatched training, parent, or data freeze")
    shard_hashes = {r["id"]: r["sha256"] for r in receipts["games"] if r["split"]=="holdout"}
    if set(shard_hashes) != {g["id"] for g in cards}:
        raise ValueError("holdout receipt coverage changed")
    dataset, hashes = public("holdout", cards, args.data, shard_hashes)
    metrics = json.loads(metrics_file.read_bytes())
    if (metrics["weights"]["npz_sha256"] != weights["npz_sha256"]
            or metrics["weights"]["agent_sha256"] != weights["agent_sha256"]
            or metrics["results"]["holdout"]["shards"] != hashes):
        raise ValueError("the reference metrics used other weights or holdout roots")
    parent, _ = model(WEIGHTS/"cash_v3.npz", weights["recipe"]["parent_sha256"])
    student, _ = model(WEIGHTS/weights["npz"], weights["npz_sha256"])
    if digest((WEIGHTS/weights["agent"]).read_bytes()) != weights["agent_sha256"]:
        raise ValueError("frozen single deployment package changed")
    active = (dataset["y"]>=0) & (dataset["legal"].sum(-1)>=2)
    rows, slots = np.nonzero(active)
    target = dataset["y"][rows, slots]
    before = predict(quantized(parent), dataset)[rows, slots].argmax(1)
    after = predict(quantized(student), dataset)[rows, slots].argmax(1)
    exact_all = accuracy(target, before, after)
    reference = metrics["results"]["holdout"]["int8"]["all"]
    for check, source in (("slots","slots"),("parent_correct","parent_correct"),("student_correct","candidate_correct"),
                          ("prediction_changed","changed"),("corrected","corrected"),("damaged","damaged")):
        if exact_all[check] != reference[source]:
            raise ValueError(f"all-slot prediction check failed: {check}")
    records = {g["gid"]:g for g in cards}
    gids = np.array([records[int(g)]["id"] for g in dataset["gid"][rows]])
    seats = np.array([records[int(g)]["seat"] for g in dataset["gid"][rows]])
    coverage = json.loads(audit_file.read_bytes())["action_opportunities"]["holdout"]
    subsets = {}
    for name, (face, response) in openings(dataset, rows, slots).items():
        group = summarize(face, response, rows, gids, seats, target, before, after)
        if group["opportunities"] != coverage[name+"_opportunity"] or group["expert_used_response"] != coverage[name+"_demonstrated"]:
            raise ValueError(f"opportunity definition differs from data audit: {name}")
        subsets[name] = group
    out = {
        "scope": "Two single-network int8-quantized parameter sets evaluated by the same CPU PyTorch decoder on recorded Master6 prefixes. Diagnostic chosen after training; checks agreement with demonstrated decisions, not rewards, forced continuations, or realized tactical success. Categories overlap.",
        "low_support_rule": "Descriptive flag if fewer than 30 opportunities, fewer than five teacher-yes or teacher-no examples, or fewer than five contributing maps.",
        "weights": {"parent": weights["recipe"]["parent_sha256"], "student": weights["npz_sha256"], "student_pack": weights["agent_sha256"]},
        "inputs_sha256": {p.name: digest(p.read_bytes()) for p in (split_file,receipts_file,weights_file,audit_file,metrics_file,Path(__file__),HERE/"train.py",ROOT/"code/cnet2_pack.py")},
        "holdout_wdl": dict(Counter(g["wdl"] for g in cards)), "roots": len(dataset["y"]),
        "all_actionable": exact_all, "groups": subsets,
    }
    write_json(args.out,out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("/workspace/sapling-master6-actor-data"))
    parser.add_argument("--out",type=Path,default=HERE/"opportunity_metrics.json")
    run(parser.parse_args())
