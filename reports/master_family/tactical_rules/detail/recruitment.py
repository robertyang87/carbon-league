"""Discretionary team composition, replacement, and two-base recruitment choices."""
from collections import Counter, defaultdict
from functools import lru_cache
from statistics import mean, median

COLLECTOR = "COLLECTOR"
PLANTER = "PLANTER"
RECRUITS = {"RECCOLLECTOR": COLLECTOR, "RECPLANTER": PLANTER}

COMPARISONS = {
    "missing_planter_minus_already_two": ("hire_planter_no_planter", "hire_planter_two_plus"),
    "missing_collector_minus_already_two": ("hire_collector_no_collector", "hire_collector_two_plus"),
    "two_free_slots_cash120_plus_minus_cash60_to89": ("double_cash120plus", "double_cash60to89"),
    "after_visible_planter_loss_minus_no_loss": ("replace_planter_after_loss", "replace_planter_no_loss"),
    "after_visible_collector_loss_minus_no_loss": ("replace_collector_after_loss", "replace_collector_no_loss"),
}


@lru_cache(maxsize=4)
def rings(size):
    near = []
    for cell in range(size * size):
        row, col = divmod(cell, size)
        near.append({cell, ((row - 1) % size) * size + col, ((row + 1) % size) * size + col,
                     row * size + (col - 1) % size, row * size + (col + 1) % size})
    return near


def submitted_destination(pos, move, size):
    row, col = divmod(pos, size)
    dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}.get(move, (0, 0))
    return ((row + dr) % size) * size + (col + dc) % size


def observe(replay, seat, key):
    cfg = replay["configuration"]
    if ([cfg["recCollectorCost"], cfg["recPlanterCost"]] != [30, 30]
            or cfg["recCollectorIncreaseCost"] or cfg["recPlanterIncreaseCost"]
            or cfg["workerLimit"] != 10):
        raise ValueError("Unexpected recruitment rules")
    nearest = rings(cfg["size"])
    steps = replay["steps"]
    counts = defaultdict(lambda: [0, 0])
    examples = {}

    # All these contrasts are restricted to midgame, visible states, and the
    # ability to purchase at least one of EITHER role. Owners share a treasury.
    for t in range(30, min(200, len(steps) - 1)):
        previous = steps[t - 1][0]["observation"]["players"][seat][2]
        players = steps[t][0]["observation"]["players"]
        cash, bases, workers, _ = players[seat]
        n = len(workers)
        if n >= 10 or cash < 30:
            continue
        base_ids = sorted(bases)
        if len(base_ids) != 2:
            raise ValueError("Expected two bases")
        moves = steps[t + 1][seat]["action"] or {}
        bought = [(i, RECRUITS[moves[b]]) for i, b in enumerate(base_ids) if moves.get(b) in RECRUITS]
        roles = {role for _, role in bought}
        composition = Counter(w[2] for w in workers.values())
        lost = {row[2] for wid, row in previous.items() if wid not in workers}
        opportunities = min(10 - n, int(cash // 30), 2)
        if len(bought) > opportunities:
            raise ValueError(f"{key} {seat} {t}: orders exceed shared resources")

        def note(label, hit):
            cell = counts[label]
            cell[0] += int(hit)
            cell[1] += 1
            ex = label + ("_yes" if hit else "_no")
            if ex not in examples:
                examples[ex] = {"key": key, "seat": seat, "step": t, "cash": cash,
                                "workers": dict(composition), "lost_roles": sorted(lost),
                                "orders": bought}

        if 2 <= n <= 8:
            # Conditional probability of ordering the role, over all admissible
            # team states (including waiting and multi-order decisions).
            for role in (COLLECTOR, PLANTER):
                nr = composition[role]
                if nr == 0:
                    note("hire_" + role.lower() + "_no_" + role.lower(), role in roles)
                elif nr >= 2:
                    note("hire_" + role.lower() + "_two_plus", role in roles)

        # Conditional replacement within a tighter composition/control stratum:
        # exactly one surviving worker of this role, both roles alive, one
        # affordable hire OR more. Opponent, map, cash and exact team size vary.
        if n >= 2 and composition[COLLECTOR] and composition[PLANTER]:
            for role in (COLLECTOR, PLANTER):
                if composition[role] == 1 and (lost == {role} or not lost):
                    suffix = "_after_loss" if lost else "_no_loss"
                    note("replace_" + role.lower() + suffix, role in roles)

        positions = [bases[b] for b in base_ids]
        own_cells = {w[0] for w in workers.values()}
        enemy = players[1 - seat][2]
        enemy_cells = {w[0] for w in enemy.values()}
        both_empty = all(pos not in own_cells | enemy_cells for pos in positions)
        under_threat = [bool(nearest[p] & enemy_cells) for p in positions]
        if 2 <= n <= 8 and cash >= 60 and both_empty and not any(under_threat):
            # Both roles and BOTH base purchases are affordable in every band.
            if cash < 90:
                note("double_cash60to89", len(bought) == 2)
            elif cash >= 120:
                note("double_cash120plus", len(bought) == 2)

        if len(bought) != 1 or not both_empty:
            continue
        chosen = bought[0][0]
        if sum(under_threat) == 1:
            note("single_hire_at_enemy_reachable_base", under_threat[chosen])
            other_base = 1 - chosen
            danger_base = chosen if under_threat[chosen] else other_base
            reaching_enemies = [w for w in enemy.values() if w[0] in nearest[positions[danger_base]]]
            tie_possible = any(w[2] == COLLECTOR and w[1] <= 0 for w in reaching_enemies)
            note("single_hire_at_threat_with_zero_collector" if tie_possible else
                 "single_hire_at_threat_without_zero_collector", under_threat[chosen])
            if under_threat[chosen]:
                note("exposed_hire_is_collector", bought[0][1] == COLLECTOR)
                enemy_actions = steps[t + 1][1 - seat]["action"] or {}
                enemy_arrived = any(submitted_destination(w[0], enemy_actions.get(wid), cfg["size"])
                                    == positions[chosen] for wid, w in enemy.items())
                note("exposed_hire_enemy_commands_same_cell", enemy_arrived)
        elif not any(under_threat):
            loaded = {pos for pos, cargo, role in workers.values() if role == COLLECTOR and cargo > 0}
            traffic = [bool(nearest[p] & loaded) for p in positions]
            if sum(traffic) == 1:
                note("single_hire_at_loaded_teammate_side", traffic[chosen])

            # Own submitted moves are used ONLY to check joint-action consistency.
            # Both bases were empty before actions. If a loaded collector plans
            # to enter one this step, does the submitted recruit use the other?
            entering = set()
            size = cfg["size"]
            for wid, (pos, cargo, role) in workers.items():
                if role != COLLECTOR or cargo <= 0:
                    continue
                dst = submitted_destination(pos, moves.get(wid), size)
                if dst in positions:
                    entering.add(positions.index(dst))
            if len(entering) == 1:
                note("single_hire_at_incoming_loaded_teammate", chosen in entering)
    return {"counts": dict(counts), "examples": examples}


def model_summary(rows, labels):
    out = {}
    for label in labels:
        counts = [r["observation"]["counts"].get(label, (0, 0)) for r in rows]
        hit = sum(h for h, _ in counts)
        total = sum(n for _, n in counts)
        rates = [h/n for h, n in counts if n]
        out[label] = {"hit": hit, "n": total, "appearances": len(rates),
                      "pooled_rate": hit/total if total else None,
                      "mean_game_rate": mean(rates) if rates else None}
    contrast = {}
    for name, (first, second) in COMPARISONS.items():
        diffs = []
        for row in rows:
            cells = row["observation"]["counts"]
            h1, n1 = cells.get(first, (0, 0))
            h2, n2 = cells.get(second, (0, 0))
            if n1 and n2:
                diffs.append(h1/n1 - h2/n2)
        contrast[name] = {"both_conditions_appearances": len(diffs),
                          "higher": sum(d > 0 for d in diffs),
                          "equal": sum(d == 0 for d in diffs),
                          "lower": sum(d < 0 for d in diffs),
                          "median_difference": median(diffs) if diffs else None}
    examples = {}
    for row in rows:
        for label, ex in row["observation"]["examples"].items():
            examples.setdefault(label, ex)
    return {"conditions": out, "within_game_contrasts": contrast, "examples": examples}


def summarize(rows):
    labels = sorted({label for r in rows for label in r["observation"]["counts"]})
    models = sorted({r["model"] for r in rows})
    return {
        "scope": "Decision steps 30–199 inclusive, price per role 30, workforce <10 and cash >=30. A row is a team decision. Outcomes are submitted orders before simultaneous settlement.",
        "definitions": {
            "hire": "2–8 total existing workers. Did the team order at least one worker of the specified role when none vs at least two of that role remain? Waiting included.",
            "replace": "Both roles alive, exactly one remaining worker of the given role, team 2–9. Compare an immediately preceding disappearance of existing worker IDs of only that role vs no disappearance. This cannot detect a newborn killed in its first transition. A replacement may be ordered in addition to another type.",
            "double": "2–8 existing workers, both bases empty, no enemy can reach either in one step. Order at both bases when 60 <= cash < 90 vs cash >=120. Both groups can afford both orders.",
            "single": "Exactly one recruit order, both bases empty. Under threat = at least one visible enemy worker in the cross around the base. Reach does not imply attack. Zero-cargo collector would tie a sole newborn collector if it moves there. For exposed hires, also measure selected hire role and the opposing submitted arrival after the decision (co-arrival is not intent). Loaded teammate side uses current positions; incoming uses actual same team's joint action at that decision (not enemy actions or next-state input).",
            "comparison": "Within-game contrasts use only appearances with opportunities in both conditions. Means/medians across games do not control for progression, opponent response or state distribution. Counts are repeated steps.",
        },
        "models": {model: model_summary([r for r in rows if r["model"] == model], labels)
                   for model in models},
    }
