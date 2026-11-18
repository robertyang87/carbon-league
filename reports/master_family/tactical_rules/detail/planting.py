"""Diagnose a planter's choice using only the state its seat can observe at decision time.

Realization is read separately from the post-action owner table. The previously
identified 1,056-game ladder cohort stays in the diagnostic pool.
"""
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
from statistics import mean


@lru_cache(maxsize=None)
def neighbors(size):
    moves = []
    for pos in range(size * size):
        r, c = divmod(pos, size)
        moves.append({
            "UP": ((r - 1) % size) * size + c, "RIGHT": r * size + (c + 1) % size,
            "DOWN": ((r + 1) % size) * size + c, "LEFT": r * size + (c - 1) % size,
        })
    return moves


CHECKS = {
    "P1_visible_carbon": {
        "a": "plant.poor4", "b": "plant.rich4",
        "eligibility": "At t30..199, planter on non-base cell with no tree of either side, opening cash covers current planting price, no enemy worker within one step.",
        "contrast": "Four orthogonal neighbors' mean visible carbon <10 vs >=30. Planter itself makes all four visible.",
        "choice": "Stay on this cell.", "realization": "Stays, survives, creates its own new tree here (new ID and creator verified)."},
    "P2_time_left": {
        "a": "plant.mid", "b": "plant.last20",
        "eligibility": "Planter on empty non-base site, cash covers price, no enemy worker within one step; mean visible carbon on four neighbors >=10.",
        "contrast": "t100..199 vs 1..20 remaining transitions (t279..298 in 300-state games).",
        "choice": "Stay on this cell.", "realization": "Stays, survives, creates own new tree here."},
    "P3_cash_buffer": {
        "a": "plant.tight", "b": "plant.buffer",
        "eligibility": "Same quiet, affordable, empty non-base site; t30..199 and mean adjacent visible carbon >=10.",
        "contrast": "Seat's pre-action cash / per-planter market price in [1, 2) vs >=4.",
        "choice": "Stay on this cell.", "realization": "Stays, survives, creates own new tree here."},
    "P4_invasion": {
        "a": "raid.approach_clear", "b": "raid.approach_guarded",
        "eligibility": "Planter off enemy trees; one adjacent enemy tree in total, no other worker currently on target; treasury could pay seizeCost. Tree & match each have >=10 further transitions.",
        "contrast": "No enemy worker can reach target in one step vs an enemy collector can reach target in one step. Excludes cases with only enemy planters nearby.",
        "choice": "Move onto that tree.", "realization": "The planter survives the step and occupies the target. Ownership can only transfer after a subsequent stay."},
    "P5_seize_or_escape": {
        "a": "raid.stay_clear", "b": "raid.stay_guarded",
        "eligibility": "Planter already on living enemy tree, pre-action cash >= seizeCost; at least 10 further turns in tree and game.",
        "contrast": "No enemy worker within one step vs enemy collector within one step. Enemy action still unknown.",
        "choice": "Stay on that enemy tree.", "realization": "Stays, survives and that same tree ID is on our side next step."},
    "P6_tree_age": {
        "a": "raid.young", "b": "raid.expiring",
        "eligibility": "Planter already on enemy tree, cash covers seizure, no enemy worker in one step, at least 30 transitions left in match.",
        "contrast": "Tree has >=30 vs 1..5 further transitions before removal after aging.",
        "choice": "Stay on that enemy tree.", "realization": "Stays, survives and the same tree ID belongs to us next step."},
}


def observe(replay, seat, key):
    """Per-appearance numerator/denominators, indexed by fixed observable conditions."""
    cfg, steps = replay["configuration"], replay["steps"]
    move = neighbors(cfg["size"])
    life = cfg["treeLifespan"]
    counts = defaultdict(lambda: [0, 0, 0])

    example = None
    strata = [name for rule in CHECKS.values() for name in (rule["a"], rule["b"])]
    witness = strata[int.from_bytes(sha256(key.encode()).digest()[:2], "big") % len(strata)]

    def add(name, selected, realized):
        nonlocal example
        if name == witness and example is None:
            example = {"key": key, "seat": seat, "t": t, "planter": uid, "cell": pos,
                       "stratum": name, "cash": own[0], "plant_price": round(price, 4),
                       "action": action or "NONE", "selected": bool(selected), "realized": bool(realized)}
            target_cell = target if name.startswith("raid.approach") else pos
            example.update(target_cell=target_cell, steps_left=tleft,
                           current_tree_id_and_age=enemy_trees.get(target_cell),
                           own_visible_neighbor_carbon=[visible[n] for n in move[pos].values()],
                           enemy_collector_can_reach_target=target_cell in collector_reach)

        row = counts[name]
        row[0] += 1
        row[1] += selected
        row[2] += realized

    for t, (frame, following) in enumerate(zip(steps, steps[1:])):
        obs = frame[0]["observation"]
        own, foe = obs["players"][seat], obs["players"][1-seat]
        planters = [(uid, w[0]) for uid, w in own[2].items() if w[2] == "PLANTER"]
        if not planters:
            continue
        after = following[0]["observation"]
        ours_after = after["players"][seat]
        actual = following[seat]["action"] or {}
        visible = frame[seat]["observation"]["carbon"]
        tleft = cfg["episodeSteps"] - 1 - t
        bases = set(own[1].values()) | set(foe[1].values())
        own_cells = {p for p, age in own[3].values()}
        enemy_trees = {p: (uid, age) for uid, (p, age) in foe[3].items()}
        existing = set(own[3]) | set(foe[3])
        births = {
            p: after["trees"][tid][0]
            for tid, (p, age) in ours_after[3].items() if tid not in existing
        }
        own_workers = {w[0] for w in own[2].values()}
        enemy_workers = {w[0] for w in foe[2].values()}
        enemy_reach = set()
        collector_reach = set()
        for p, _, kind in foe[2].values():
            reach = (p, *move[p].values())
            enemy_reach.update(reach)
            if kind == "COLLECTOR":
                collector_reach.update(reach)
        price = cfg["plantCost"]
        if len(own[3]) > cfg["playerTreeProtectiveNumber"]:
            n_alive = sum(age < life for player in (own, foe) for _, age in player[3].values())
            price += cfg["plantCostInflationRatio"] * cfg["plantCostInflationBase"] ** n_alive
        can_seize = own[0] >= cfg["seizeCost"]
        for uid, pos in planters:
            action = actual.get(uid)
            stayed = action is None or action == "NONE"
            dest = pos if stayed else move[pos][action]
            survived = uid in ours_after[2]
            if pos not in own_cells and pos not in enemy_trees and pos not in bases:
                planted = stayed and survived and births.get(pos) == uid
                if own[0] >= price and pos not in enemy_reach:
                    mean_carbon = sum(visible[n] for n in move[pos].values()) / 4
                    if 30 <= t < 200:
                        if mean_carbon < 10:
                            add("plant.poor4", stayed, planted)
                        elif mean_carbon >= 30:
                            add("plant.rich4", stayed, planted)
                        if mean_carbon >= 10:
                            if own[0] < 2 * price:
                                add("plant.tight", stayed, planted)
                            elif own[0] >= 4 * price:
                                add("plant.buffer", stayed, planted)
                    if mean_carbon >= 10:
                        if 100 <= t < 200:
                            add("plant.mid", stayed, planted)
                        elif 1 <= tleft <= 20:
                            add("plant.last20", stayed, planted)
            if not can_seize:
                continue
            if pos in enemy_trees:
                tid, age = enemy_trees[pos]
                left = life - age
                took = stayed and survived and tid in ours_after[3]
                if min(left, tleft) >= 10:
                    if pos not in enemy_reach:
                        add("raid.stay_clear", stayed, took)
                    elif pos in collector_reach:
                        add("raid.stay_guarded", stayed, took)
                if pos not in enemy_reach and tleft >= 30:
                    if left >= 30:
                        add("raid.young", stayed, took)
                    elif 1 <= left <= 5:
                        add("raid.expiring", stayed, took)
            elif tleft >= 10:
                targets = [p for p in move[pos].values() if p in enemy_trees]
                if len(targets) != 1:
                    continue
                target = targets[0]
                if life - enemy_trees[target][1] < 10 or target in enemy_workers or target in own_workers:
                    continue
                entered = dest == target
                arrived = entered and survived and ours_after[2][uid][0] == target
                if target not in enemy_reach:
                    add("raid.approach_clear", entered, arrived)
                elif target in collector_reach:
                    add("raid.approach_guarded", entered, arrived)
    return {"counts": dict(counts), "example": example}


def summarize(rows):
    """Support, decision rate and downstream realization for every ranked model."""
    models = sorted({row["model"] for row in rows})
    groups = {}
    for model in models:
        appearances = [row["observation"] for row in rows if row["model"] == model]
        result = {}
        for check, spec in CHECKS.items():
            names = (spec["a"], spec["b"])
            paired = []
            totals = {}
            for name in names:
                counts = [a["counts"][name] for a in appearances if name in a["counts"]]
                n, selected, realized = (sum(c[i] for c in counts) for i in range(3))
                totals[name] = {
                    "appearances_exposed": len(counts), "opportunities": n,
                    "selected": selected, "realized": realized,
                    "selection_rate": selected / n if n else None,
                    "realization_rate": realized / n if n else None,
                }
            for appearance in appearances:
                a, b = (appearance["counts"].get(n) for n in names)
                if a and b:
                    paired.append(b[1] / b[0] - a[1] / a[0])
            result[check] = {"strata": totals,
                             "within_game_b_minus_a": {
                                 "games": len(paired), "positive": sum(d > 0 for d in paired),
                                 "negative": sum(d < 0 for d in paired),
                                 "ties": sum(d == 0 for d in paired),
                                 "mean": mean(paired) if paired else None,
                             }}
        groups[model] = result
    examples = {}
    for row in rows:
        event = row["observation"]["example"]
        if event:
            key = event["stratum"] + ("_selected" if event["selected"] else "_declined")
            examples.setdefault(key, {"model": row["model"], **event})
    return {"rules": CHECKS, "by_model": groups, "examples": examples,
            "limitations": [
                "Affordability is measured before simultaneous base recruits, deposits and other planters. Do not convert it to a guarantee of settlement.",
                "Planter sees its four orthogonal neighbors. This is a partial measure of resources, not the absorption of all eight neighbors.",
                "One opportunity is one planter at one decision. Conditional rates are observational, exposure is correlated within games; within-game contrasts use only games exposing both strata.",
                "Tree approach is a motion command; the target can only be seized after a later stay. This report cross-checks its immediate arrival, not subsequent ownership.",
            ]}
