"""One-step action interventions on selected real loaded-collector escapes.

For each Top Five player take the first verified example with an actual attack
and one where the same possible attack does not happen. Fix the other submitted
actions, including both the opponent's complete action and all other friendly
workers. This deliberately selects contrasting outcomes, not a benefit estimate.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys

from cargo_defense import geometry, landing

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "code"))
from fastsim import FastCarbon


def replay_trial(replay, tick, seat, uid, stay):
    before, following = replay["steps"][tick:tick + 2]
    state = before[0]["observation"]
    if "full_carbon" not in state:
        raise ValueError("Replay oracle ground layer required to reconstruct the evaluator")
    sim = FastCarbon(replay["configuration"]).load_observation(state)
    actions = [deepcopy(record["action"] or {}) for record in following]
    if stay:
        actions[seat].pop(uid)
    sim.step(*actions)
    out = sim.full_observation()
    lost = [list(record) for record in sim.last_collisions if record[0] == uid]
    return out, {
        "worker_after": out["players"][seat][2].get(uid), "cash_by_seat_after": list(sim.cash),
        "worker_collision": lost[0] if lost else None,
    }


def sample(replay, seat, key, wanted):
    cfg = replay["configuration"]
    if cfg["moveCost"] != 0:
        raise ValueError("This collision comparator assumes the cohort's zero movement charge")
    dirs, reach = geometry(cfg["size"])
    frames = replay["steps"]
    found = {}
    for tick, (before, after) in enumerate(zip(frames, frames[1:])):
        own = before[0]["observation"]["players"][seat]
        foe = before[0]["observation"]["players"][1-seat]
        bases = set(own[1].values()) | set(foe[1].values())
        ours = after[seat]["action"] or {}
        theirs = after[1-seat]["action"] or {}
        for uid, (cell, cargo, role) in sorted(own[2].items()):
            action = ours.get(uid)
            if role != "COLLECTOR" or cargo < 100 or action not in dirs[cell] or cell in bases:
                continue
            target = dirs[cell][action]
            if target in bases or uid not in after[0]["observation"]["players"][seat][2]:
                continue
            dangers = [(eid, pos, load) for eid, (pos, load, kind) in foe[2].items()
                       if kind == "COLLECTOR" and load <= cargo]
            if any(target in reach[pos] for _, pos, _ in dangers):
                continue
            approaches = [(eid, pos, load) for eid, pos, load in dangers if cell in reach[pos]]
            if not approaches:
                continue
            # Actual simultaneous replies determine which illustrative bucket to show.
            arrived = [eid for eid, (pos, _, _) in foe[2].items()
                       if landing(pos, theirs.get(eid), dirs) == cell]
            family = "attacked_if_stay" if arrived else "threat_not_realized"
            if family not in wanted or family in found:
                continue
            out, fact = replay_trial(replay, tick, seat, uid, False)
            expected = after[0]["observation"]
            if out["players"] != expected["players"] or out["step"] != expected["step"]:
                # A reconstructed observation lacks ID counters consumed by old dead
                # units. Require exact replayed player state, including all IDs.
                continue
            _, alternative = replay_trial(replay, tick, seat, uid, True)
            if family == "attacked_if_stay":
                loss = alternative["worker_collision"]
                if not loss or loss[3] != 1 - seat or loss[4] != 1 - seat:
                    continue
            elif alternative["worker_after"] is None:
                continue
            found[family] = {
                "key": key, "seat": seat, "t": tick, "worker": uid, "cell": cell,
                "cargo": cargo, "recorded_action": action, "chosen_destination": target,
                "observable_reaching_collectors": approaches, "enemy_actually_entered_old_cell": arrived,
                "original_players_and_clock_exact": True, "recorded": fact, "force_stay": alternative,
            }
        if wanted.issubset(found):
            break
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    store = HERE.parent
    table_bytes = (store / "top5_score_rows.json").read_bytes()
    table = json.loads(table_bytes)
    inventory = json.loads((store / "top5_inventory.json").read_bytes())
    table_digest = hashlib.sha256(table_bytes).hexdigest()
    if table_digest != inventory["score_rows_sha256"]:
        raise ValueError("Score table identity changed")
    result = {rank["name"]: {} for rank in inventory["ranks"]}
    kinds = {"attacked_if_stay", "threat_not_realized"}
    witnesses = {}
    for key, packet in sorted(table.items()):
        names = [packet["name_" + side] for side in packet["order"]]
        needed = {s: kinds.difference(result[n]) for s, n in enumerate(names) if n in result and len(result[n]) < 2}
        if not needed:
            continue
        blob = (args.replays / (key + ".json.gz")).read_bytes()
        replay = json.loads(gzip.decompress(blob))
        file_hash = hashlib.sha256(blob).hexdigest()
        ref = inventory["replays"][key]
        if file_hash != ref["sha256"]:
            canonical = json.dumps(replay, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            if hashlib.sha256(canonical.encode()).hexdigest() != ref["canonical_sha256"]:
                raise ValueError(f"Unexpected replay bytes {key}")
        if replay["rewards"] != [packet["scores"][s] for s in packet["order"]]:
            raise ValueError(f"Score/seat identity mismatch {key}")
        for seat, wanted in needed.items():
            added = sample(replay, seat, key, wanted)
            result[names[seat]].update(added)
            if added:
                witnesses[key] = ref["canonical_sha256"]
        if all(len(v) == 2 for v in result.values()):
            break
    if any(set(v) != kinds for v in result.values()):
        raise ValueError("Could not establish both examples for all five players")
    output = {
        "ranking_score_table_sha256": table_digest,
        "simulator_sha256": hashlib.sha256((HERE.parents[3] / "code" / "fastsim.py").read_bytes()).hexdigest(),
        "selection": "First case per player/bucket by sorted match key, decision t and worker ID, with original next players and clock exactly reproduced. Cargo >=100, no base at source or chosen destination. At least one visible opposing collector of equal or smaller load can reach the original cell, none can reach the selected destination. Attacked bucket also requires the governing collision to transfer the load to a surviving enemy collector if we stay; negative control requires nobody on the opposing side actually chose that cell and the staying worker survives.",
        "intervention": "Change this collector's recorded move to NONE. Keep all other submitted simultaneous actions fixed. Oracle ground carbon only restores the simulator after selection of the public-position threat. Entry worker list format is [cell, cargo, role]. A collision row is [losing ID, owner, cargo, cargo recipient, winning collector owner]. Endpoints are the next step only; not closed-loop estimates.",
        "replay_canonical_sha256": witnesses, "examples": result,
    }
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
