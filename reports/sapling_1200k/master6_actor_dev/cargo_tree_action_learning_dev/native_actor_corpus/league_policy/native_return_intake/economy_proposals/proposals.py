"""Training-only proposals over exact physical team orders of the frozen 851c actor.

Call after the acting seat's public observation has advanced the package's Feat.
Only the twelve agreed turns have proposal distributions. No RNG is consumed
unless a caller explicitly samples an order under its own generator.
"""

import math
import sys

import numpy as np


CATEGORIES = ("recruit", "plant_or_capture", "freight", "ordinary")
STEPS = frozenset(range(12, 277, 24))
CLASS_PROBABILITY = 0.25
CANDIDATES_TO_CHECK = 4
WITNESSES_TO_KEEP = 2


def command_key(command):
    return tuple(sorted(command.items()))


def memberships(model, public, slot, best, move):
    """Classify a legal native alternative in physical board coordinates."""
    feature = model._FEAT
    geometry = sys.modules["cnet2_feat"]
    seat = int(public["player"])
    friend = public["players"][seat]
    enemy = public["players"][1 - seat]
    uid = feature.own_uid[slot]
    tags = {"ordinary": ["ordinary"]}
    context = {"uid": uid}
    if slot >= 10:
        base_actions = geometry.BASE_ACTS
        tags["recruit"] = [f"{base_actions[best]}_to_{base_actions[move]}"]
        return tags, context

    pos, cargo, role = friend[2][uid]
    pos, cargo = int(pos), float(cargo)
    context.update(position=pos, cargo=cargo, role=role)
    home = list(map(int, friend[1].values()))
    if role == "PLANTER" and 0 in (best, move):
        any_base = set(home).union(map(int, enemy[1].values()))
        own_trees = {int(tree[0]) for tree in friend[3].values()}
        if pos not in any_base and pos not in own_trees:
            enemy_trees = {int(tree[0]) for tree in enemy[3].values()}
            target = "capture" if pos in enemy_trees else "plant"
            direction = "enter_stay" if move == 0 else "leave_stay"
            tags["plant_or_capture"] = [target, direction]
            context["public_cash"] = float(friend[0])
    if role != "COLLECTOR" or cargo <= 0:
        return tags, context

    def destination(action):
        physical_action = int(geometry.CANON_MOVE[action]) if seat else action
        return int(geometry.DEST[pos, physical_action])

    destinations = [destination(best), destination(move)]
    on_base = [cell in home for cell in destinations]
    distance = [min((int(geometry.DIST[cell, base]) for base in home), default=None)
                for cell in destinations]
    enemies = [int(worker[0]) for worker in enemy[2].values()
               if worker[2] == "COLLECTOR"]
    threats = [sum(int(geometry.DIST[cell, foe]) <= 1 for foe in enemies)
               for cell in destinations]
    freight_tags = []
    if on_base[0] != on_base[1]:
        freight_tags.append("enter_home" if on_base[1] else "leave_home")
    if distance[0] != distance[1]:
        freight_tags.append("nearer_home" if distance[1] < distance[0] else "farther_home")
    if threats[0] != threats[1]:
        freight_tags.append("fewer_adjacent_foes" if threats[1] < threats[0]
                            else "more_adjacent_foes")
    if freight_tags:
        tags["freight"] = freight_tags
        context.update(destinations=destinations, home_distance=distance, nearby_foes=threats)
    return tags, context


def validated_alternate(model, x, aux, base, base_command, trace, route, slot, move):
    labels, alt_trace, alt_route = model._sample_branch(x, aux, (slot, move))
    if alt_route != route:
        return None, "changed_decoder_route"
    if int(labels[slot]) != move:
        return None, "forced_move_missing"
    if not np.array_equal(labels[:slot], base[:slot]):
        return None, "changed_earlier_prefix"
    if slot not in alt_trace or not all(np.array_equal(a, b)
                                       for a, b in zip(trace[slot], alt_trace[slot])):
        return None, "changed_selected_slot_distribution"
    if model._return_conflicts(x, labels) - model._return_conflicts(x, base):
        return None, "new_team_conflict"
    cmd = model._physical(labels)
    if cmd == base_command:
        return None, "same_physical_command"
    return cmd, None


def alternative_probability(best_score, other_score):
    # Stable two-action softmax, mixed equally with a fair coin.
    difference = other_score - best_score
    return 0.25 + 0.5 * math.exp(-float(np.logaddexp(0.0, -difference)))


def build(model, x, aux, public):
    """Return the complete distribution, or None on ordinary unscheduled turns."""
    if int(public["step"]) not in STEPS:
        return None
    if "full_carbon" in public or int(public["player"]) != model._FEAT.seat:
        raise ValueError("expected the acting seat's public observation")
    if model._FEAT.step != int(public["step"]) or model._capture is not None or model._forced is not None:
        raise ValueError("stale features or nested package decoding")
    # The original _sample_branch sets this diagnostic even when no action is
    # ultimately issued. Keep this offline calculation transparent to its caller.
    old_route = getattr(model, "_chosen_route", None)
    old_counts = model._COUNTS
    model._COUNTS = {role: detail.copy() for role, detail in old_counts.items()}
    try:
        return _build(model, x, aux, public)
    finally:
        model._chosen_route = old_route
        model._COUNTS = old_counts


def _build(model, x, aux, public):
    base, trace, route = model._sample_branch(x, aux)
    base_command = model._physical(base)
    candidates = []
    grouped = {name: [] for name in CATEGORIES}
    for slot, (_, scores, legal) in sorted(trace.items()):
        slot = int(slot)
        best = int(base[slot])
        if not legal[best]:
            raise ValueError("baseline selected an illegal action")
        for move in map(int, np.flatnonzero(legal)):
            if move == best:
                continue
            tags, context = memberships(model, public, slot, best, move)
            gap = float(scores[best]) - float(scores[move])
            record = {"slot": slot, "best": best, "move": move, "memberships": tags,
                      "gap": gap if math.isfinite(gap) else None, "context": context}
            candidates.append(record)
            for name in tags:
                grouped[name].append(record)

    # Some classes share the same witness; force that branch only once.
    cache = {}
    classes = {}
    orders = {}

    def add_path(cmd, mass, category, source, slot=None, move=None):
        key = command_key(cmd)
        entry = orders.setdefault(key, {"command": cmd, "probability": 0.0, "paths": []})
        entry["probability"] += mass
        entry["paths"].append({"class": category, "source": source, "mass": mass,
                               "slot": slot, "move": move})

    for name, records in grouped.items():
        ranked = sorted(records, key=lambda r: (
            math.inf if r["gap"] is None else r["gap"], r["slot"], r["move"]))
        attempted, witnesses, seen_commands = [], [], set()
        for record in ranked[:CANDIDATES_TO_CHECK]:
            slot, move = record["slot"], record["move"]
            key = (slot, move)
            if record["gap"] is None:
                cmd, reason = None, "nonfinite_native_gap"
            else:
                if key not in cache:
                    cache[key] = validated_alternate(
                        model, x, aux, base, base_command, trace, route, slot, move)
                cmd, reason = cache[key]
            if cmd is not None and command_key(cmd) in seen_commands:
                reason = "duplicate_physical_command_within_class"
            attempted.append({"slot": slot, "move": move, "status": reason or "accepted"})
            if reason is not None:
                continue
            seen_commands.add(command_key(cmd))
            _, scores, legal = trace[slot]
            best = record["best"]
            logits = [float(scores[best]), float(scores[move])]
            p_alt = alternative_probability(*logits)
            witnesses.append({"slot": slot, "actions": [best, move], "tags": record["memberships"][name],
                              "physical_context": record["context"], "legal": legal.tolist(),
                              "prefix": base[:slot].tolist(), "route": bool(route),
                              "logits": logits, "command": cmd, "conditional_p_alt": p_alt})
            if len(witnesses) == WITNESSES_TO_KEEP:
                break

        if not witnesses:
            add_path(base_command, CLASS_PROBABILITY, name, "empty_fallback")
        for witness in witnesses:
            weight = CLASS_PROBABILITY / len(witnesses)
            p_alt = witness["conditional_p_alt"]
            slot, move = witness["slot"], witness["actions"][1]
            add_path(base_command, weight * (1.0 - p_alt), name, "baseline", slot, move)
            add_path(witness["command"], weight * p_alt, name, "alternate", slot, move)
        classes[name] = {"eligible_count": len(records), "attempted": attempted,
                         "witnesses": witnesses, "fallback": not witnesses,
                         "untried_count": len(records) - len(attempted)}
    result = list(orders.values())
    if not all(0.0 < row["probability"] <= 1 for row in result) or not math.isclose(
            math.fsum(row["probability"] for row in result), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("invalid physical command probabilities")
    return {"baseline": base_command, "baseline_probability": orders[command_key(base_command)]["probability"],
            "orders": result, "classes": classes, "eligible_witnesses": candidates,
            "labels": {"immediate_economic_effect": None, "official_terminal_effect": None}}


def draw(menu, rng):
    """Sample a physical action and return its full marginal behavior probability."""
    if menu is None:
        raise ValueError("ordinary turns use the existing agent")
    rows = menu["orders"]
    chosen = rows[int(rng.choice(len(rows), p=[r["probability"] for r in rows]))]
    return dict(chosen["command"]), chosen["probability"]
