"""Measure one-step threats to owned trees across the frozen Top 2 diagnostic corpus.

Run from any checkout containing the frozen Git objects. Pass --public-dir with the
92 files recovered by deep_review/recover_public.py. Only players, rules and the
clock are used to recognize threats. Next-frame actions and owners score outcomes.
"""
import argparse
import csv
import gzip
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent / "deep_review"
REV = "37965ea578851b5f3ff88ffc42d7119ff716b5cc"
MODELS = {"claude-AlphaCarbon-Master5s": "Master5s", "claude-AlphaCarbon-Master6": "Master6"}
DELTAS = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
CATEGORIES = ("on_tree", "adjacent_only", "outside_one_step")


@lru_cache(maxsize=None)
def moves(size):
    table = []
    for cell in range(size * size):
        row, col = divmod(cell, size)
        table.append({act: ((row + dr) % size) * size + (col + dc) % size
                      for act, (dr, dc) in DELTAS.items()})
    return table


def read_replay(row, public_dir):
    if row["source"].startswith("https://"):
        raw = (public_dir / (row["key"] + ".json.gz")).read_bytes()
    else:
        raw = subprocess.check_output(["git", "-C", str(REVIEW), "show", REV + ":" + row["source"]])
    if hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError("Replay identity mismatch: " + row["key"])
    return json.loads(gzip.decompress(raw))


def new_totals():
    return {"games": 0, "exposure": Counter(),
            "reachable_tree_steps": {k: Counter() for k in CATEGORIES}, "outcomes": Counter()}


def observe(steps, seat, config, model, game, opponent, total, examples):
    directions = moves(config["size"])
    exposure, outcomes = total["exposure"], total["outcomes"]
    total["games"] += 1
    local = Counter()
    for tick, (frame, following) in enumerate(zip(steps, steps[1:])):
        obs = frame[0]["observation"]
        post = following[0]["observation"]
        if (obs["step"], post["step"]) != (tick, tick + 1):
            raise ValueError("Unexpected step alignment")
        own, other = obs["players"][seat], obs["players"][1 - seat]
        own_after, enemy_after = post["players"][seat], post["players"][1 - seat]
        own_actions = following[seat]["action"] or {}
        enemy_actions = following[1 - seat]["action"] or {}
        own_at = {pos for pos, _, _ in own[2].values()}
        defendable = set()
        arrivings = defaultdict(list)
        for wid, (pos, _, role) in own[2].items():
            defendable.update(directions[pos].values())
            action = own_actions.get(wid)
            dest = directions[pos][action] if action else pos
            arrivings[dest].append((wid, role, action))
        planters = {pos: wid for wid, (pos, _, role) in other[2].items() if role == "PLANTER"}
        adjacent = {dest for pos in planters for dest in directions[pos].values()}
        for tree, (cell, age) in own[3].items():
            if age >= config["treeLifespan"]:
                continue
            captured = tree in enemy_after[3]
            if captured and cell not in planters:
                raise ValueError(f"Capture without preceding enemy planter on tree {game}:{tick}:{tree}")
            threat = "on_tree" if cell in planters else "adjacent_only" if cell in adjacent else "outside_one_step"
            near = cell not in own_at and cell in defendable
            entered = bool(arrivings.get(cell))
            if near:
                counts = total["reachable_tree_steps"][threat]
                counts["opportunities"] += 1
                counts["ally_entered_cell"] += int(entered)
            if threat != "on_tree":
                continue
            enemy = planters[cell]
            stayed = not enemy_actions.get(enemy)
            if tree not in own_after[3] and not captured:
                raise ValueError("Non-expiring tree disappeared")
            if captured and (not stayed or entered):
                raise ValueError("Unexpected capture despite attacker moving or existing ally contest")
            exposure["tree_steps"] += 1
            exposure["enemy_stayed"] += int(stayed)
            exposure["captured"] += int(captured)
            exposure["enemy_cash_pre_below_fee"] += int(other[0] < config["seizeCost"])
            exposure["ally_entered_tree"] += int(entered)
            local["enemy_on_tree"] += 1
            local["capture"] += int(captured)
            local["ally_contest"] += int(entered)
            name = ("entered" if entered else "no_entry") + ("_enemy_stayed" if stayed else "_enemy_moved")
            outcomes[name + "_total"] += 1
            outcomes[name + "_captured"] += int(captured)
            # Give one observed example per distinct case, in stable fixture order.
            if entered and stayed:
                case = "ally_contested_enemy_stayed"
            elif entered and not stayed:
                case = "ally_entered_enemy_moved_same_turn"
            elif captured and near:
                case = "enemy_seized_with_ally_in_reach"
            elif not entered and not stayed and near:
                case = "no_entry_enemy_left_tree_intact"
            else:
                continue
            key = model + ":" + case
            if key not in examples:
                examples[key] = {"game": game, "opponent": opponent, "seat": seat, "step": tick,
                                 "tree_id": tree, "age": age, "cell": cell, "model": model,
                                 "enemy_planter": enemy, "enemy_cash_pre": other[0],
                                 "enemy_action": enemy_actions.get(enemy) or "NONE",
                                 "ally_arrivals": arrivings.get(cell, []), "captured": captured,
                                 "tree_owner_after": 1 - seat if captured else seat}
    for metric in ("enemy_on_tree", "capture", "ally_contest"):
        exposure["games_with_" + metric] += int(local[metric] > 0)
    return local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with (REVIEW / "all_games.csv").open() as source:
        rows = list(csv.DictReader(source))
    matches = json.loads((REVIEW / "inputs" / "official_matches.json").read_text())
    if len(rows) != len(matches) or len(matches) != 352 or len({r["key"] for r in rows}) != len(rows):
        raise ValueError("Unexpected frozen cohort")
    totals = {name: new_totals() for name in MODELS.values()}
    examples = {}
    for row in sorted(rows, key=lambda r: r["key"]):
        match = matches[row["key"]]
        replay = read_replay(row, args.public_dir)
        order = match["order"]
        if replay["rewards"] != [match["scores"][s] for s in order]:
            raise ValueError("Fixture scores mismatch")
        for seat, side in enumerate(order):
            model = MODELS.get(match["name_" + side])
            if model:
                local = observe(replay["steps"], seat, replay["configuration"], model, row["key"],
                                match["name_" + order[1 - seat]], totals[model], examples)
                expected = row["opponent_seized"] if seat == int(row["seat"]) else row["seized"]
                if local["capture"] != int(expected):
                    raise ValueError(f"Capture count differs from the independent game card: {row['key']}")
    if [v["games"] for v in totals.values()] != [191, 163]:
        raise ValueError("Unexpected seat totals")
    receipt = {"games": len(rows), "frozen_replay_tree": REV, "totals_by_model": totals, "examples": examples,
               "units": "A tree-step is one live tree in one decision; multiple workers/games/steps are correlated.",
               "reachable": "Before either action, no allied worker on this tree and at least one existing ally adjacent. Arrivals count commanded destinations irrespective of survival. A newly recruited worker cannot move to a tree in the same step.",
               "exposure": "An enemy planter stands on a non-expiring owned tree; actual capture requires survival, staying and enough cash at resolution. No requirement on pre-action cash; an earlier deposit/recruit may change it.",
               "scope": "Descriptive defense evidence for all matches; no inference of underlying intent, no training or model selection. Tree captures independently checked against the frozen game cards."}
    args.out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
