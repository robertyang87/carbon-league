"""One-step cargo defense and interception on the historical Top Five heavy-board snapshot.

Called by mine.py. Only positions, roles, loads, own bases and public clock before
actions enter the conditions. Following submissions label destinations; the next
state checks whether a worker actually survived. No hidden carbon map is read.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from math import inf
from statistics import mean


@lru_cache(maxsize=None)
def geometry(size):
    directions = {}
    reach = {}
    for cell in range(size * size):
        r, c = divmod(cell, size)
        destinations = {
            "UP": ((r - 1) % size) * size + c,
            "RIGHT": r * size + (c + 1) % size,
            "DOWN": ((r + 1) % size) * size + c,
            "LEFT": r * size + (c - 1) % size,
        }
        directions[cell] = destinations
        reach[cell] = (cell, *destinations.values())
    return directions, reach


def landing(pos, action, directions):
    if action in (None, "NONE"):
        return pos
    return directions[pos][action]


def observe(replay, seat, key):
    cfg = replay["configuration"]
    if cfg["moveCost"] != 0:
        raise ValueError("Arrival-load comparisons require moveCost=0 for this cohort")
    directions, reach = geometry(cfg["size"])
    counts = defaultdict(Counter)
    examples = {}
    frames = replay["steps"]
    for tick, (before, after) in enumerate(zip(frames, frames[1:])):
        players = before[0]["observation"]["players"]
        own, enemy = players[seat], players[1 - seat]
        ours, theirs = own[2], enemy[2]
        home, their_bases = set(own[1].values()), set(enemy[1].values())
        own_pos = {w[0] for w in ours.values()}
        enemy_pos = {w[0] for w in theirs.values()}
        # An enemy collector died in a tie too. Minimum accounts for every opposing worker
        # that can reach the square; it is not a prediction that it will choose to do so.
        min_cargo = {}
        enemy_reach = set()
        for cell, load, role in theirs.values():
            enemy_reach.update(reach[cell])
            if role == "COLLECTOR":
                for land in reach[cell]:
                    if load < min_cargo.get(land, inf):
                        min_cargo[land] = load
        viable_victims = {cell: (vid, load) for vid, (cell, load, role) in theirs.items()
                          if role == "COLLECTOR" and load > 0 and cell not in home | their_bases
                          and min_cargo[cell] > 0}
        own_moves = after[seat]["action"] or {}
        enemy_moves = after[1 - seat]["action"] or {}
        post = after[0]["observation"]["players"]
        enemy_arrived = set()
        real_min = {}
        for eid, (cell, load, role) in theirs.items():
            target = landing(cell, enemy_moves.get(eid), directions)
            enemy_arrived.add(target)
            if role == "COLLECTOR" and load < real_min.get(target, inf):
                real_min[target] = load
        own_lands = Counter(landing(v[0], own_moves.get(uid), directions) for uid, v in ours.items())

        for wid, (pos, cargo, role) in ours.items():
            if role != "COLLECTOR":
                continue
            dest = landing(pos, own_moves.get(wid), directions)
            cells = reach[pos]
            if cargo == 0:
                available = [c for c in cells[1:] if c in viable_victims]
                if len(available) == 1:
                    prey = available[0]
                    victim, amount = viable_victims[prey]
                    label = "chase_100plus" if amount >= 100 else "chase_under30" if amount < 30 else "chase_30to100"
                    event = counts[label]
                    event["n"] += 1
                    entered = dest == prey
                    stayed = not enemy_moves.get(victim) or enemy_moves.get(victim) == "NONE"
                    event["selected"] += entered
                    event["prey_stayed_when_chased"] += entered and stayed
                    capture = entered and stayed and victim not in post[1-seat][2] and wid in post[seat][2]
                    event["prey_removed_and_hunter_survived"] += capture
                    if capture and label == "chase_100plus" and "interception" not in examples:
                        examples["interception"] = {"t": tick, "actor": wid, "actor_from": pos, "cell": prey,
                                                    "victim": victim, "cargo": amount}
                continue

            # The five-cell neighborhood may contain a newly recruited enemy on its base.
            # Restrict to neighborhoods disjoint from enemy bases so a public spawn
            # cannot invalidate the collision classification. A free alternative means
            # no currently resident teammate (except the actor), not a guaranteed
            # reservation. Actual teammate co-arrivals are reported separately.
            if any(c in their_bases for c in cells):
                continue
            safe = [c for c in cells if (c == pos or c not in own_pos) and min_cargo.get(c, inf) > cargo]
            if not safe:
                continue
            lethal_chosen = min_cargo.get(dest, inf) <= cargo
            lethal_real = real_min.get(dest, inf) <= cargo
            if lethal_real and wid in post[seat][2]:
                raise ValueError(f"Collector survived a simultaneous equal/lighter enemy: {key}, {tick}, {wid}")
            if min_cargo.get(pos, inf) <= cargo:
                choices = counts["stay_exposed_choose_safe"]
                choices["n"] += 1
                choices["selected"] += not lethal_chosen
                choices["selected_free_safe"] += dest in safe
                choices["moved"] += dest != pos
                choices["remained_exposed"] += lethal_chosen
                choices["actual_equal_or_lighter_arrivals"] += lethal_real
                choices["died_any_cause"] += wid not in post[seat][2]
                choices["other_own_worker_same_destination"] += own_lands[dest] > 1
                if not lethal_chosen and "escape" not in examples:
                    examples["escape"] = {"t": tick, "actor": wid, "cargo": cargo, "cell": pos, "chosen": dest,
                                          "enemy_min_if_stay": min_cargo[pos], "safe_alternatives": len(safe)}

            adjacent = home.intersection(cells[1:])
            if pos in home or len(adjacent) != 1:
                continue
            depot = next(iter(adjacent))
            if depot in own_pos or depot in enemy_pos or not any(c != depot for c in safe):
                continue
            if cargo < 100:
                continue
            minimum = min_cargo.get(depot, inf)
            if minimum <= cargo:
                danger = "lethal_collector"
            elif minimum < inf:
                danger = "only_heavier_collectors"
            elif depot in enemy_reach:
                danger = "planter_only"
            else:
                danger = "no_enemy"
            stage = "last10" if tick >= cfg["episodeSteps"] - 11 else "before_last10"
            for band in ("all", stage):
                event = counts[f"home_{danger}_{band}"]
                event["n"] += 1
                entered = dest == depot
                event["selected"] += entered
                event["actual_enemy_arrivals_when_entered"] += entered and dest in enemy_arrived
                event["actual_equal_or_lighter_when_entered"] += entered and lethal_real
                event["disappeared_when_entered"] += entered and wid not in post[seat][2]
                event["ally_arrival_when_entered"] += entered and own_lands[dest] > 1
            if danger == "lethal_collector" and dest != depot and "risk_diversion" not in examples:
                examples["risk_diversion"] = {"t": tick, "actor": wid, "cargo": cargo, "cell": pos, "home": depot,
                                               "enemy_min_at_home": minimum, "chosen": dest, "stayed": pos == dest}
    return {"counts": {k: dict(v) for k, v in counts.items()}, "examples": examples}


def summarize(rows):
    contrasts = [
        ("enter_home_lethal_minus_clear", "home_lethal_collector_all", "home_no_enemy_all"),
        ("enter_home_planter_minus_lethal", "home_planter_only_all", "home_lethal_collector_all"),
        ("enter_home_heavier_minus_lethal", "home_only_heavier_collectors_all", "home_lethal_collector_all"),
        ("chase_heavy_minus_light", "chase_100plus", "chase_under30"),
        ("enter_lethal_home_deadline_minus_earlier", "home_lethal_collector_last10",
         "home_lethal_collector_before_last10"),
    ]
    by_model = {}
    for model in dict.fromkeys(row["model"] for row in rows):
        appearances = [r for r in rows if r["model"] == model]
        totals, per_match, examples = defaultdict(Counter), {}, {}
        for row in appearances:
            metrics = row["observation"]["counts"]
            for name, cell in metrics.items():
                totals[name].update(cell)
                totals[name]["games"] += 1
            for tag, case in row["observation"]["examples"].items():
                if tag not in examples:
                    examples[tag] = {"match": row["key"], "seat": row["seat"], **case}
        for tag, first, second in contrasts:
            diffs = []
            for row in appearances:
                cells = row["observation"]["counts"]
                a, b = cells.get(first), cells.get(second)
                if a and b:
                    diffs.append(a["selected"] / a["n"] - b["selected"] / b["n"])
            per_match[tag] = {"games_with_both": len(diffs),
                              "mean_paired_rate_difference": mean(diffs) if diffs else None,
                              "negative": sum(d < 0 for d in diffs), "positive": sum(d > 0 for d in diffs)}
        by_model[model] = {"appearances": len(appearances), "cells": {k: dict(v) for k, v in totals.items()},
                           "paired_game_contrasts": per_match, "examples": examples}
    return {
        "units": "Current worker-turn; selected = observed action, including workers that die on that turn. "
                 "Actions for state t come from frame t+1; no hidden carbon. Each count also reports games with exposure.",
        "rules": {
            "stay_exposed_choose_safe": "Loaded collector; staying reachable by enemy collector with cargo <= ours; "
                "at least one potential non-lethal alternative with no current ally there; no enemy base in neighborhood. "
                "Selected means any chosen non-lethal landing; selected_free_safe additionally excludes ally-occupied cells. "
                "Actual allied arrivals exclude recruits. Reach includes an opponent staying or moving one cell.",
            "home": "Cargo >=100, off base and exactly one home adjacent, home currently empty; at least one other "
                "unoccupied-by-other-allies and non-lethal alternative; neighborhood disjoint from enemy bases. "
                "Selected = actually commanded into home. Stratify by minimum cargo of any enemy collector which "
                "could arrive there (zero moveCost), or planter-only reach, or no enemy can arrive. Last10 = t289..298.",
            "chase": "Empty collector adjacent to exactly one viable positive-cargo enemy collector on a non-base "
                "square. No enemy zero-cargo collector can counterreach the target (a tie for minimum kills all; "
                "only a unique lightest collector survives). Selected = command into the opponent's current square. A recorded "
                "capture requires that enemy stayed, target disappeared and attacking collector survived. This does "
                "not label a worker's intention or success elsewhere after its target moves.",
        },
        "by_model": by_model,
    }
